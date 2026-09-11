"""Typed handoffs between the three review agents.

The schemas are the point. Three agents chained on free text is three prompts
in a row; three agents chained on a validated structure is a pipeline, because
each stage can be checked, retried and tested against a stub in isolation.

Every stage also carries a ``confidence`` and each agent is told to abstain
rather than guess, so a low-confidence stage is visible in the trace instead of
being laundered into a fluent recommendation.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

TOPICS = Literal[
    "allocation", "concentration", "currency", "liquidity", "cash", "goal", "mandate"
]
SEVERITY = Literal["low", "medium", "high"]


class Finding(BaseModel):
    """One material observation, traceable to a computed fact."""

    fact_key: str = Field(description="The [key] of the computed fact this rests on")
    topic: TOPICS
    observation: str = Field(
        description="One sentence in plain language, quoting the figure from the fact"
    )
    materiality: SEVERITY


class Findings(BaseModel):
    items: list[Finding] = Field(min_length=1, max_length=8)
    not_material: list[str] = Field(
        default_factory=list,
        description="Fact keys deliberately excluded as immaterial",
    )
    confidence: float = Field(ge=0.0, le=1.0)


class Risk(BaseModel):
    fact_key: str
    severity: SEVERITY
    rationale: str = Field(description="Why this matters for THIS client specifically")
    policy_basis: str = Field(
        description="The source label, e.g. S2, whose rule this is measured against, "
        "or 'none' if no retrieved source covers it"
    )


class RiskAssessment(BaseModel):
    risks: list[Risk] = Field(min_length=1, max_length=8)
    overall: str = Field(description="Two sentences on the client's overall risk posture")
    confidence: float = Field(ge=0.0, le=1.0)


class Action(BaseModel):
    action: str = Field(description="A specific, checkable thing the client can do")
    rationale: str
    addresses: list[str] = Field(description="fact_keys this action resolves")
    sources: list[str] = Field(
        default_factory=list, description="Source labels cited, e.g. ['S1', 'S3']"
    )


class Recommendation(BaseModel):
    summary: str = Field(description="Two or three sentences the client reads first")
    actions: list[Action] = Field(min_length=1, max_length=6)
    caveats: str = Field(
        description="What this review could not establish, and what needs an adviser"
    )
    confidence: float = Field(ge=0.0, le=1.0)


class Review(BaseModel):
    """The full pipeline result, kept so a recommendation can be reconstructed."""

    client_id: str
    findings: Findings
    assessment: RiskAssessment
    recommendation: Recommendation
    sources: list[str] = Field(default_factory=list)
