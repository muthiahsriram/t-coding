import json
from datetime import date

import pytest

from app.agents import facts, pipeline
from app.agents.schemas import Findings, Recommendation, RiskAssessment
from app.rag.corpus import build_chunks
from app.rag.retriever import Retriever
from app.seed import get_client

TODAY = date(2026, 1, 1)


# --- the deterministic half ------------------------------------------------


def test_every_fact_carries_the_threshold_it_was_measured_against():
    """A finding of '25% in one holding' is meaningless without the limit."""
    for fact in facts.portfolio_facts(get_client("C002"), TODAY):
        assert any(ch.isdigit() for ch in fact.statement)


def test_breaches_are_reported_before_compliant_facts():
    measured = facts.portfolio_facts(get_client("C002"), TODAY)
    breaches = [i for i, f in enumerate(measured) if f.breach]
    compliant = [i for i, f in enumerate(measured) if not f.breach]

    assert not breaches or not compliant or max(breaches) < min(compliant)


def test_a_position_over_the_single_name_limit_is_a_breach():
    client = get_client("C002")
    holdings = client.all_holdings()
    invested = sum(h.market_value for h in holdings)
    largest = max(
        (h.instrument.ticker for h in holdings),
        key=lambda t: sum(h.market_value for h in holdings if h.instrument.ticker == t),
    )
    weight = sum(h.market_value for h in holdings if h.instrument.ticker == largest) / invested

    fact = next(
        f for f in facts.concentration(client) if f.key == f"concentration-{largest}"
    )
    assert fact.breach == (round(weight, 4) > facts.SINGLE_NAME_LIMIT)


def test_drift_is_measured_against_the_clients_own_model_portfolio():
    """A conservative client must not be measured against the growth model."""
    conservative = get_client("C004")
    statements = " ".join(f.statement for f in facts.drift(conservative))

    assert "conservative model weight" in statements


def test_every_seeded_client_produces_facts_without_raising():
    from app.seed import BOOK

    for client in BOOK.values():
        assert facts.portfolio_facts(client, TODAY)


# --- the pipeline ----------------------------------------------------------


def _findings_json(client) -> str:
    key = facts.portfolio_facts(client, TODAY)[0].key
    return json.dumps(
        {
            "items": [
                {
                    "fact_key": key,
                    "topic": "concentration",
                    "observation": "A single holding is 25.0% of invested assets.",
                    "materiality": "high",
                }
            ],
            "not_material": [],
            "confidence": 0.9,
        }
    )


_ASSESSMENT_JSON = json.dumps(
    {
        "risks": [
            {
                "fact_key": "concentration-VWRA.L",
                "severity": "high",
                "rationale": "Above the 15% single-name limit.",
                "policy_basis": "S1",
            }
        ],
        "overall": "Concentrated but solvent. No immediate liquidity problem.",
        "confidence": 0.8,
    }
)

_RECOMMENDATION_JSON = json.dumps(
    {
        "summary": "One position has grown past the house limit.",
        "actions": [
            {
                "action": "Trim the largest holding back toward its model weight.",
                "rationale": "It exceeds the 15% single-name limit.",
                "addresses": ["concentration-VWRA.L"],
                "sources": ["S1"],
            }
        ],
        "caveats": "Your monthly expenses are not on file, so the reserve is unverified.",
        "confidence": 0.75,
    }
)


class _ScriptedModel:
    """Replays queued responses and records every prompt it was sent."""

    def __init__(self, *responses: str):
        self.responses = list(responses)
        self.prompts: list[list[dict]] = []

    async def __call__(self, messages, **kwargs):
        self.prompts.append(messages)
        return self.responses.pop(0)


@pytest.fixture
def retriever():
    return Retriever(build_chunks())


@pytest.fixture
def scripted(monkeypatch):
    def install(*responses):
        model = _ScriptedModel(*responses)
        monkeypatch.setattr(pipeline, "complete", model)
        return model

    return install


async def test_the_three_stages_run_in_order_and_each_sees_the_previous_output(
    scripted, retriever
):
    """If a stage cannot see the one before it, this is not a pipeline."""
    client = get_client("C002")
    model = scripted(_findings_json(client), _ASSESSMENT_JSON, _RECOMMENDATION_JSON)

    review = await pipeline.run_review(client, retriever, today=TODAY)

    assert len(model.prompts) == 3
    analyst, risk, advisor = (p[0]["content"] for p in model.prompts)
    assert "analyst" in analyst.lower()
    assert "risk officer" in risk.lower()
    assert "adviser" in advisor.lower()

    # Stage 2 received stage 1's structured output, stage 3 received stage 2's.
    assert "COMPUTED FACTS" in model.prompts[0][1]["content"]
    assert "FINDINGS" in model.prompts[1][1]["content"]
    assert "RISK ASSESSMENT" in model.prompts[2][1]["content"]
    assert review.recommendation.actions[0].addresses == ["concentration-VWRA.L"]


async def test_the_analyst_never_has_to_compute_a_number(scripted, retriever):
    """Every figure reaches the model precomputed; that is where accuracy lives."""
    client = get_client("C002")
    model = scripted(_findings_json(client), _ASSESSMENT_JSON, _RECOMMENDATION_JSON)

    await pipeline.run_review(client, retriever, today=TODAY)

    prompt = model.prompts[0][1]["content"]
    for fact in facts.portfolio_facts(client, TODAY):
        assert fact.statement in prompt


async def test_invalid_json_is_retried_with_the_validation_error_fed_back(
    scripted, retriever
):
    client = get_client("C002")
    model = scripted(
        '{"items": [], "confidence": 5}',  # empty list and out-of-range confidence
        _findings_json(client),
        _ASSESSMENT_JSON,
        _RECOMMENDATION_JSON,
    )

    review = await pipeline.run_review(client, retriever, today=TODAY)

    assert len(model.prompts) == 4  # one retry
    retry = model.prompts[1][-1]["content"]
    assert "did not validate" in retry
    assert "confidence" in retry  # the specific failing field, not a bare re-ask
    assert review.findings.items


async def test_a_stage_that_fails_twice_raises_rather_than_inventing_output(
    scripted, retriever
):
    scripted("not json", "still not json")

    with pytest.raises(Exception):
        await pipeline.run_review(get_client("C002"), retriever, today=TODAY)


async def test_prose_around_the_json_is_tolerated(scripted, retriever):
    """Models fence and preface JSON more often than the prompt implies."""
    client = get_client("C002")
    model = scripted(
        f"Here you go:\n```json\n{_findings_json(client)}\n```",
        _ASSESSMENT_JSON,
        _RECOMMENDATION_JSON,
    )

    await pipeline.run_review(client, retriever, today=TODAY)

    assert len(model.prompts) == 3  # no retry needed


async def test_a_finding_citing_a_nonexistent_fact_is_dropped(scripted, retriever):
    """An uncited finding is exactly the hallucinated number this design avoids."""
    client = get_client("C002")
    real = facts.portfolio_facts(client, TODAY)[0].key
    payload = json.dumps(
        {
            "items": [
                {
                    "fact_key": real,
                    "topic": "concentration",
                    "observation": "Real.",
                    "materiality": "high",
                },
                {
                    "fact_key": "invented-fact",
                    "topic": "goal",
                    "observation": "Made up.",
                    "materiality": "high",
                },
            ],
            "not_material": [],
            "confidence": 0.9,
        }
    )
    scripted(payload, _ASSESSMENT_JSON, _RECOMMENDATION_JSON)

    review = await pipeline.run_review(client, retriever, today=TODAY)

    assert [i.fact_key for i in review.findings.items] == [real]


async def test_the_risk_stage_only_retrieves_policy_and_regulation(
    scripted, retriever
):
    """Scoring a breach against a market commentary note is not a policy basis."""
    client = get_client("C002")
    model = scripted(_findings_json(client), _ASSESSMENT_JSON, _RECOMMENDATION_JSON)

    await pipeline.run_review(client, retriever, today=TODAY)

    context = model.prompts[1][1]["content"]
    assert "commentary" not in context.lower().split("policy and regulation")[1][:400]


async def test_every_stage_is_reported_for_the_trace(scripted, retriever):
    client = get_client("C002")
    scripted(_findings_json(client), _ASSESSMENT_JSON, _RECOMMENDATION_JSON)
    seen: list[str] = []

    async def report(stage, status, detail):
        seen.append(stage)

    await pipeline.run_review(client, retriever, today=TODAY, report=report)

    assert seen == ["Analyst", "Risk", "Advisor"]


async def test_the_rendered_review_shows_only_the_client_facing_stage(
    scripted, retriever
):
    client = get_client("C002")
    scripted(_findings_json(client), _ASSESSMENT_JSON, _RECOMMENDATION_JSON)

    rendered = pipeline.format_review(
        await pipeline.run_review(client, retriever, today=TODAY)
    )

    assert "One position has grown past the house limit." in rendered
    assert "Trim the largest holding" in rendered
    # Internal machinery must not leak into the client's view.
    assert "confidence" not in rendered.lower()
    assert "fact_key" not in rendered


def test_the_coded_thresholds_match_the_policy_documents():
    """facts.py duplicates the limits as code; the two must not drift apart."""
    from app.rag.corpus import load_handwritten

    policy = next(
        d.text for d in load_handwritten() if d.meta.doc_id == "internal-concentration-limits"
    )
    assert f"{facts.SINGLE_NAME_LIMIT:.0%}".replace("%", "%") in policy
    assert f"{facts.SECTOR_LIMIT:.0%}" in policy
    assert f"{facts.CASH_DRAG_LIMIT:.0%}" in policy

    models = next(
        d.text for d in load_handwritten() if d.meta.doc_id == "internal-model-portfolios"
    )
    assert "5 percentage points" in models and "10 points" in models
