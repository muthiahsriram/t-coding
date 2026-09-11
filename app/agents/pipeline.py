"""The sequential portfolio review: Analyst -> Risk -> Advisor.

Sequential because the dependency is real. The risk agent cannot score a
finding the analyst did not measure, and the advisor cannot propose an action
without a ranked risk to justify it - reorder the stages and the pipeline stops
making sense, which is the test of whether a workflow is genuinely staged or
just three prompts in a row.

Each stage:

* receives the previous stage's *validated* output, never its raw text;
* gets its own system prompt, its own retrieval slice and its own schema;
* is retried once with the validation error fed back when the JSON is wrong.

Only the advisor's output is ever shown to the client. The first two stages
exist so that what the advisor says is grounded in something checkable, and
they are surfaced in the UI as trace rather than as answers.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Awaitable, Callable
from datetime import date
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from app.agents.facts import portfolio_facts, render_facts
from app.agents.schemas import Findings, Recommendation, Review, RiskAssessment
from app.domain import Client
from app.llm import complete
from app.portfolio import summarise
from app.rag.documents import Chunk
from app.rag.retriever import Filters, Retriever, format_context

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# A step reporter: stage name, status line, detail. Chainlit passes one that
# renders nested steps; tests and scripts pass nothing.
Reporter = Callable[[str, str, str], Awaitable[None]]

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _extract_json(text: str) -> str:
    """Pull a JSON object out of a model response.

    Models fence JSON, or preface it, more often than the prompt would suggest.
    Failing the whole stage on a stray "Here you go:" would burn a retry on
    something with an obvious deterministic fix.
    """
    fenced = _FENCE.search(text)
    if fenced:
        return fenced.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return text[start : end + 1]
    return text.strip()


async def _structured(
    system: str, user: str, schema: type[T], *, stage: str
) -> tuple[T, int]:
    """Call the model and validate into ``schema``, retrying once on failure.

    Returns the parsed object and the number of attempts, so the trace can show
    that a retry happened rather than hiding it.
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    for attempt in (1, 2):
        raw = await complete(messages)
        try:
            return schema.model_validate_json(_extract_json(raw)), attempt
        except (ValidationError, ValueError) as exc:
            log.warning("%s: attempt %d failed validation: %s", stage, attempt, exc)
            if attempt == 2:
                raise
            # Feed the error back rather than re-asking blind: the model
            # corrects a named field far more reliably than a repeated prompt.
            messages += [
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": (
                        "That did not validate against the required schema.\n"
                        f"{exc}\n\n"
                        "Return corrected JSON only. No prose, no code fence."
                    ),
                },
            ]

    raise AssertionError("unreachable")


def _schema_instruction(schema: type[BaseModel]) -> str:
    return (
        "Reply with a single JSON object and nothing else. It must validate "
        "against this JSON Schema:\n"
        f"{json.dumps(schema.model_json_schema(), indent=None)}"
    )


# --- stage 1: the analyst --------------------------------------------------

ANALYST_SYSTEM = """You are the analyst on a Singapore wealth advisory desk.

You are given a client briefing and a list of facts that have ALREADY been
computed from the client's actual holdings. Your job is to decide which facts
are material for this client and state each in one plain sentence.

Rules:
- Never compute a figure. Every number you write must appear verbatim in a fact.
- Never invent a fact_key. Use the [key] exactly as given.
- Breaches are material by default. A non-breach is material only if it changes
  how a breach should be read.
- Do not recommend anything. That is not your stage.
- Set confidence below 0.6 if the facts are too thin to support a review."""


async def analyse(client: Client, today: date) -> tuple[Findings, list[str], int]:
    facts = portfolio_facts(client, today)
    user = (
        f"CLIENT BRIEFING\n{summarise(client, today)}\n\n"
        f"COMPUTED FACTS\n{render_facts(facts)}\n\n"
        f"{_schema_instruction(Findings)}"
    )
    findings, attempts = await _structured(
        ANALYST_SYSTEM, user, Findings, stage="analyst"
    )

    # The model can still cite a key that does not exist. Dropping unknown keys
    # is safer than trusting them: a finding with no fact behind it is exactly
    # the hallucinated number this design exists to prevent.
    known = {fact.key for fact in facts}
    kept = [item for item in findings.items if item.fact_key in known]
    dropped = [item.fact_key for item in findings.items if item.fact_key not in known]
    if dropped:
        log.warning("analyst cited unknown fact keys, dropped: %s", dropped)
    if kept:
        findings = findings.model_copy(update={"items": kept})

    return findings, [fact.render() for fact in facts], attempts


# --- stage 2: the risk agent -----------------------------------------------

RISK_SYSTEM = """You are the risk officer on a Singapore wealth advisory desk.

You receive findings from the analyst and extracts from the firm's policies and
the applicable regulations. Score each finding for THIS client.

Rules:
- Severity reflects consequence for this client, not the size of the number. A
  10-point drift on a goal fifteen years out is lower severity than a currency
  mismatch on a goal in two years.
- Cite the source label (S1, S2, ...) whose rule you measured against. If no
  retrieved source covers a finding, write "none" - do not invent a policy.
- Never rely on a source marked withdrawn or superseded.
- Do not propose actions. That is not your stage."""

RISK_QUERY = (
    "concentration limits, currency policy for near-term goals, liquidity and "
    "emergency reserve, drift and rebalancing thresholds, mandate mismatch"
)


async def assess(
    client: Client, findings: Findings, retriever: Retriever, today: date
) -> tuple[RiskAssessment, list[Chunk], int]:
    sources = await retriever.search(
        RISK_QUERY,
        client=client,
        k=6,
        filters=Filters(doc_types=("policy", "regulation")),
        today=today,
    )
    user = (
        f"CLIENT: {client.name}, {client.risk_profile} mandate, age {client.age}\n\n"
        f"FINDINGS\n{findings.model_dump_json(indent=2)}\n\n"
        f"POLICY AND REGULATION\n{format_context(sources)}\n\n"
        f"{_schema_instruction(RiskAssessment)}"
    )
    assessment, attempts = await _structured(RISK_SYSTEM, user, RiskAssessment, stage="risk")
    return assessment, sources, attempts


# --- stage 3: the advisor --------------------------------------------------

ADVISOR_SYSTEM = """You are the adviser speaking to the client directly.

You receive scored risks and reference material. Turn them into specific things
this client can do, in plain language.

Rules:
- Order actions the way the firm's policy orders them: restore the emergency
  reserve, cure concentration breaches, correct a mandate mismatch, fund goals
  that are behind, then rebalance drift.
- Every action names the fact_keys it addresses and the source labels it rests on.
- Never state or imply a forecast of a return, price or index level.
- Never instruct a trade as though it were executed. You propose; the client and
  their adviser decide.
- Say plainly in caveats what this review could not establish.
- You are addressing the client. Write to them, not about them."""


async def advise(
    client: Client,
    assessment: RiskAssessment,
    retriever: Retriever,
    today: date,
) -> tuple[Recommendation, list[Chunk], int]:
    query = " ".join(risk.rationale for risk in assessment.risks) or RISK_QUERY
    sources = await retriever.search(query, client=client, k=6, today=today)
    user = (
        f"CLIENT: {client.name}, {client.risk_profile} mandate, age {client.age}, "
        f"monthly surplus {client.monthly_surplus:,.0f} SGD\n\n"
        f"RISK ASSESSMENT\n{assessment.model_dump_json(indent=2)}\n\n"
        f"REFERENCE\n{format_context(sources)}\n\n"
        f"{_schema_instruction(Recommendation)}"
    )
    recommendation, attempts = await _structured(
        ADVISOR_SYSTEM, user, Recommendation, stage="advisor"
    )
    return recommendation, sources, attempts


# --- the pipeline ----------------------------------------------------------


async def run_review(
    client: Client,
    retriever: Retriever,
    *,
    today: date | None = None,
    report: Reporter | None = None,
) -> Review:
    """Run all three stages and return the assembled review."""
    today = today or date.today()

    async def emit(stage: str, status: str, detail: str) -> None:
        if report is not None:
            await report(stage, status, detail)

    findings, facts, attempts = await analyse(client, today)
    await emit(
        "Analyst",
        f"{len(findings.items)} material finding(s) from {len(facts)} computed facts"
        f"{'' if attempts == 1 else f' (retried {attempts - 1}x)'}",
        "\n".join(f"- **{i.fact_key}** ({i.materiality}) {i.observation}" for i in findings.items)
        + f"\n\nConfidence {findings.confidence:.2f}",
    )

    assessment, policy_sources, attempts = await assess(client, findings, retriever, today)
    await emit(
        "Risk",
        f"{len(assessment.risks)} risk(s) scored against "
        f"{len(policy_sources)} policy source(s)"
        f"{'' if attempts == 1 else f' (retried {attempts - 1}x)'}",
        "\n".join(
            f"- **{r.fact_key}** — {r.severity} — {r.rationale} _(basis: {r.policy_basis})_"
            for r in assessment.risks
        )
        + f"\n\n{assessment.overall}\n\nConfidence {assessment.confidence:.2f}",
    )

    recommendation, reference_sources, attempts = await advise(
        client, assessment, retriever, today
    )
    await emit(
        "Advisor",
        f"{len(recommendation.actions)} action(s) proposed"
        f"{'' if attempts == 1 else f' (retried {attempts - 1}x)'}",
        f"Confidence {recommendation.confidence:.2f}",
    )

    seen: list[str] = []
    for chunk in [*policy_sources, *reference_sources]:
        if chunk.meta.title not in seen:
            seen.append(chunk.meta.title)

    return Review(
        client_id=client.client_id,
        findings=findings,
        assessment=assessment,
        recommendation=recommendation,
        sources=seen,
    )


def format_review(review: Review) -> str:
    """The client-facing rendering. Only the advisor's stage is shown."""
    lines = [
        "## Portfolio review",
        "",
        review.recommendation.summary,
        "",
        "### What to do",
    ]
    for i, action in enumerate(review.recommendation.actions, start=1):
        cited = f" _{', '.join(action.sources)}_" if action.sources else ""
        lines.append(f"{i}. **{action.action}**  \n{action.rationale}{cited}")

    lines += ["", "### Worth knowing", review.recommendation.caveats]
    if review.sources:
        lines += ["", "### Sources", *(f"- {title}" for title in review.sources)]
    return "\n".join(lines)
