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
    measured = {f.key: f for f in facts.concentration(client)}

    for holding in holdings:
        ticker = holding.instrument.ticker
        weight = sum(
            h.market_value for h in holdings if h.instrument.ticker == ticker
        ) / invested
        fact = measured.get(f"concentration-{ticker}")
        if fact is not None:
            assert fact.breach == (round(weight, 4) > facts.SINGLE_NAME_LIMIT)


def test_the_single_name_limit_applies_to_equities_not_bonds():
    """Applied to everything, it reports the conservative model's own 60% bond
    allocation as a permanent breach of the firm's policy."""
    conservative = get_client("C004")
    bonds = {
        h.instrument.ticker
        for h in conservative.all_holdings()
        if h.instrument.asset_class == "bond"
    }

    flagged = {
        f.key.removeprefix("concentration-") for f in facts.concentration(conservative)
    }
    assert bonds and not (flagged & bonds)


def test_drift_is_measured_against_the_clients_own_model_portfolio():
    """A conservative client must not be measured against the growth model."""
    conservative = get_client("C004")
    statements = " ".join(f.statement for f in facts.drift(conservative))

    assert "conservative model weight" in statements


def test_drift_is_measured_by_asset_class_not_by_instrument():
    """The rebalancing policy defines drift over asset classes. Per-instrument,
    a client who swapped one global tracker for another drifts twice while
    their allocation never moved."""
    keys = {f.key for f in facts.drift(get_client("C002"))}

    assert keys <= {"drift-equity", "drift-bond", "drift-commodity"}


def test_the_book_actually_drifts():
    """Regression guard. The seeded holdings were once generated *from* the
    model weights, so actual equalled target to the last decimal and this whole
    branch of the review could never fire."""
    from app.seed import BOOK

    drifting = [c for c in BOOK.values() if facts.drift(c)]
    assert len(drifting) >= 4


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
        # Copied: the retry path extends the same list in place, so holding the
        # reference would make every recorded prompt for a stage identical.
        self.prompts.append(list(messages))
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
    scripted(_findings_json(client), _ASSESSMENT_JSON, _RECOMMENDATION_JSON)

    calls: list[dict] = []
    original = retriever.search

    async def spy(query, **kwargs):
        calls.append(kwargs)
        return await original(query, **kwargs)

    retriever.search = spy
    try:
        await pipeline.run_review(client, retriever, today=TODAY)
    finally:
        retriever.search = original

    risk_filters = calls[0]["filters"]
    assert risk_filters.doc_types == ("policy", "regulation")

    sources = await original(
        pipeline.RISK_QUERY, client=client, k=6, filters=risk_filters, today=TODAY
    )
    assert sources
    assert all(c.meta.doc_type in ("policy", "regulation") for c in sources)


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
    """facts.py restates the limits as code so the review can compute against
    them. Nothing stops the two copies drifting apart except this test."""
    from app.rag.corpus import load_handwritten

    kb = {d.meta.doc_id: d.text for d in load_handwritten()}

    limits = kb["internal-concentration-limits"]
    assert f"{facts.SINGLE_NAME_LIMIT:.0%}" in limits
    assert f"{facts.SINGLE_NAME_WATCH:.0%}" in limits
    assert f"{facts.SECTOR_LIMIT:.0%}" in limits
    assert f"{facts.HOME_BIAS_LIMIT:.0%}" in limits
    assert f"{facts.CASH_DRAG_LIMIT:.0%}" in limits

    models = kb["internal-model-portfolios"]
    assert f"{facts.DRIFT_REPORT * 100:.0f} percentage points" in models
    assert f"{facts.DRIFT_ACTION * 100:.0f} points" in models


def test_the_coded_model_allocations_match_the_policy_document():
    """The four models are described in prose in the KB and as numbers in the
    seed. A client rebalanced toward a model the document does not describe is
    the worst kind of wrong: internally consistent and unjustifiable."""
    from app.rag.corpus import load_handwritten
    from app.seed import MODEL_ALLOCATIONS

    models = next(
        d.text for d in load_handwritten() if d.meta.doc_id == "internal-model-portfolios"
    )

    for profile, weights in MODEL_ALLOCATIONS.items():
        sentence = next(s for s in models.split("\n") if s.startswith(f"The {profile} model"))
        assert abs(sum(weights.values()) - 1.0) < 1e-9
        for bucket, weight in weights.items():
            if weight:
                assert f"{weight:.0%}" in sentence, (profile, bucket)
