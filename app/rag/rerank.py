"""A domain reranker for wealth advice.

Off-the-shelf rerankers maximise the probability that a chunk answers the
query. That is the right objective when every wrong answer costs the same. It
is the wrong objective here, because the costs are wildly asymmetric: a
withdrawn note describing a CPF technique that stopped working in 2025 is
*textually perfect* for a query about that technique, and acting on it is a
compliance incident. Similarity has to be overridable.

So the reranker multiplies the fused retrieval score by four signals that a
cross-encoder cannot see, because they live in metadata rather than in prose:

1. **Supersession** - a withdrawn document is suppressed, not merely demoted.
2. **Authority** - the regulator's text outranks our commentary about it.
3. **Recency** - rules have effective dates and the newest binding one wins.
4. **Portfolio relevance** - a factsheet for something the client actually
   holds beats one for something they don't. This is the signal no generic
   reranker can reproduce: it depends on who is asking.

Every multiplier is recorded on the chunk, so a ranking can be explained
rather than asserted.
"""

from __future__ import annotations

from datetime import date

from app.domain import Client
from app.rag.documents import Chunk

# A withdrawn document is not deleted: a client may ask why the old advice
# changed, and answering that needs the old text. Suppressing to a fifth keeps
# it reachable on an explicit query and off the top of everything else.
SUPERSEDED_PENALTY = 0.2

AUTHORITY_MULTIPLIER = {3: 1.20, 2: 1.08, 1: 1.00}

# Rules decay slowly; commentary decays fast. A three-year-old CPF regulation
# is probably still in force, a three-year-old market note is worthless.
HALF_LIFE_YEARS = {
    "regulation": 6.0,
    "policy": 4.0,
    "factsheet": 3.0,
    "faq": 3.0,
    "meeting_note": 2.0,
    "commentary": 1.0,
}
RECENCY_FLOOR = 0.65

# Kept deliberately small. These are priors about what the client is likely to
# care about, not evidence that a chunk answers the question, and at 1.3 they
# were large enough to put a factsheet for a held instrument above the document
# that actually defined the term being asked about.
HELD_TICKER_BOOST = 1.15
HELD_ASSET_CLASS_BOOST = 1.05
RISK_PROFILE_BOOST = 1.08
OWN_NOTE_BOOST = 1.12


def _recency(chunk: Chunk, today: date) -> float:
    if chunk.meta.effective_date is None:
        return 1.0
    half_life = HALF_LIFE_YEARS.get(chunk.meta.doc_type, 3.0)
    age_years = max(0.0, (today - chunk.meta.effective_date).days / 365.25)
    decay = 0.5 ** (age_years / half_life)
    return RECENCY_FLOOR + (1.0 - RECENCY_FLOOR) * decay


def _portfolio(chunk: Chunk, client: Client | None) -> float:
    if client is None:
        return 1.0

    holdings = client.all_holdings()
    tickers = {h.instrument.ticker for h in holdings}
    asset_classes = {h.instrument.asset_class for h in holdings}

    multiplier = 1.0
    if chunk.meta.ticker:
        multiplier *= HELD_TICKER_BOOST if chunk.meta.ticker in tickers else 1.0
    elif chunk.meta.asset_class in asset_classes:
        multiplier *= HELD_ASSET_CLASS_BOOST

    if client.risk_profile in chunk.meta.risk_profiles:
        multiplier *= RISK_PROFILE_BOOST
    if chunk.meta.client_id == client.client_id:
        multiplier *= OWN_NOTE_BOOST

    return multiplier


def rerank(
    candidates: list[Chunk],
    *,
    client: Client | None = None,
    today: date | None = None,
    penalise_superseded: bool = True,
) -> list[Chunk]:
    """Reorder ``candidates`` by their fused score adjusted for metadata.

    Each chunk must already carry a ``fused`` score. The returned chunks carry
    every multiplier that was applied, so the ranking is auditable.

    ``penalise_superseded`` is off when the caller explicitly asked for
    withdrawn documents. The filter has already expressed the intent, and
    applying the penalty on top would suppress the very thing that was
    requested -- the demotion exists to stop withdrawn text arriving
    *unasked*, not to make it unreachable.
    """
    today = today or date.today()
    reranked = []

    for chunk in candidates:
        supersession = (
            SUPERSEDED_PENALTY
            if (penalise_superseded and chunk.meta.superseded_by)
            else 1.0
        )
        authority = AUTHORITY_MULTIPLIER.get(chunk.meta.authority_tier, 1.0)
        recency = _recency(chunk, today)
        portfolio = _portfolio(chunk, client)

        base = chunk.scores.get("fused", 0.0)
        reranked.append(
            chunk.with_scores(
                supersession=supersession,
                authority=authority,
                recency=recency,
                portfolio=portfolio,
                reranked=base * supersession * authority * recency * portfolio,
            )
        )

    return sorted(reranked, key=lambda c: -c.scores["reranked"])


def explain(chunk: Chunk) -> str:
    """One line describing why a chunk ranked where it did."""
    parts = []
    for name, value in (
        ("superseded", chunk.scores.get("supersession", 1.0)),
        ("authority", chunk.scores.get("authority", 1.0)),
        ("recency", chunk.scores.get("recency", 1.0)),
        ("portfolio", chunk.scores.get("portfolio", 1.0)),
    ):
        if abs(value - 1.0) > 1e-9:
            parts.append(f"{name} x{value:.2f}")
    return ", ".join(parts) or "no metadata adjustment"
