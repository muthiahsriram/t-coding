"""The document model the retrieval layer indexes.

Metadata is a typed field on every chunk rather than a loose dict, because the
reranker and the access filter both read it. A typo in a dict key would fail
open — returning another client's meeting notes — so the shape is pinned here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from datetime import date

# Roughly a paragraph and a half. Small enough that a retrieved chunk is mostly
# signal, large enough that a rule and its exception stay in the same chunk --
# splitting "you may withdraw at 62" from "unless the account is pledged" is how
# a retrieval bug becomes a compliance answer that is confidently wrong.
CHUNK_WORDS = 140
CHUNK_OVERLAP_WORDS = 30

# Authority tiers. A rule published by the regulator outranks our own commentary
# about that rule; the reranker turns this into a score multiplier.
AUTHORITY_TIER = {
    "MAS": 3,
    "CPF Board": 3,
    "IRAS": 3,
    "HDB": 3,
    "issuer": 2,
    "internal": 1,
}


@dataclass(frozen=True)
class DocMeta:
    """Everything about a document that is not its prose."""

    doc_id: str
    title: str
    doc_type: str  # regulation | policy | factsheet | commentary | meeting_note | faq
    authority: str = "internal"
    language: str = "en"  # en | zh | ms | ta
    effective_date: date | None = None
    superseded_by: str | None = None
    ticker: str | None = None
    asset_class: str | None = None
    region: str | None = None
    # Set only on documents belonging to one client. This is an access
    # boundary, not a ranking signal: see ``Retriever._visible``.
    client_id: str | None = None
    risk_profiles: tuple[str, ...] = ()

    @property
    def authority_tier(self) -> int:
        return AUTHORITY_TIER.get(self.authority, 1)


@dataclass(frozen=True)
class Document:
    meta: DocMeta
    text: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    meta: DocMeta
    position: int
    # Populated by the retriever; kept here so a result is self-describing.
    scores: dict[str, float] = field(default_factory=dict)

    @property
    def citation(self) -> str:
        stamp = f", {self.meta.effective_date.isoformat()}" if self.meta.effective_date else ""
        return f"[{self.meta.title} ({self.meta.authority}{stamp})]"

    def with_scores(self, **scores: float) -> Chunk:
        return replace(self, scores={**self.scores, **scores})


_PARAGRAPH = re.compile(r"\n\s*\n")


def chunk_document(
    document: Document,
    *,
    max_words: int = CHUNK_WORDS,
    overlap: int = CHUNK_OVERLAP_WORDS,
) -> list[Chunk]:
    """Split a document into overlapping, paragraph-aligned chunks.

    Paragraph-aligned rather than a flat sliding window: these documents are
    lists of rules, and a window that starts mid-rule retrieves a fragment with
    no subject. Paragraphs longer than ``max_words`` are windowed as a fallback.
    """
    chunks: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            chunks.append(" ".join(buffer))

    for paragraph in _PARAGRAPH.split(document.text.strip()):
        words = paragraph.split()
        if not words:
            continue

        if len(words) > max_words:
            flush()
            buffer = []
            step = max(1, max_words - overlap)
            for start in range(0, len(words), step):
                window = words[start : start + max_words]
                if window:
                    chunks.append(" ".join(window))
                if start + max_words >= len(words):
                    break
            continue

        if len(buffer) + len(words) > max_words:
            flush()
            # Carry the tail forward so a rule split across the boundary is
            # still retrievable from the following chunk.
            buffer = buffer[-overlap:] if overlap else []
        buffer.extend(words)

    flush()

    return [
        Chunk(
            chunk_id=f"{document.meta.doc_id}#{i}",
            text=text,
            meta=document.meta,
            position=i,
        )
        for i, text in enumerate(chunks)
    ]
