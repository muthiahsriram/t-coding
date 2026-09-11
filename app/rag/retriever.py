"""Hybrid retrieval: BM25 + dense vectors, fused with RRF, then reranked.

The dense half is a plain NumPy matrix multiply. At ~1,100 chunks x 1,024
dimensions the index is about 4 MB of float32 and an exact search costs a
couple of milliseconds - so brute force gives *perfect* recall for less
operational cost than any approximate index, and there is nothing a vector
database would add. That stops being true somewhere around a million chunks;
the note in the README says what we would move to.

The dense half is also optional. Where the workspace serves no embedding
endpoint, ``semantic_available`` is False and the same call path returns
keyword-only results instead of failing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import numpy as np

from app.domain import Client
from app.rag.corpus import build_chunks
from app.rag.documents import Chunk
from app.rag.embeddings import DatabricksEmbedder, resolve_embedder
from app.rag.keyword import BM25Index
from app.rag.rerank import rerank

log = logging.getLogger(__name__)

# Reciprocal Rank Fusion. 60 is the value from the original paper and is not
# sensitive; what matters is that it is large enough that the top few ranks of
# each list are close in contribution, so one list cannot dominate on rank 1.
RRF_K = 60

# How deep to go in each list before fusing. Deeper than the number we return,
# so a chunk ranked mediocre by both methods can still surface by agreeing.
CANDIDATE_DEPTH = 40


@dataclass(frozen=True)
class Filters:
    """Metadata narrowing applied before scoring."""

    doc_types: tuple[str, ...] | None = None
    languages: tuple[str, ...] | None = None
    authorities: tuple[str, ...] | None = None
    include_superseded: bool = False


def _rrf(ranked_indices: list[np.ndarray], weights: list[float]) -> dict[int, float]:
    """Fuse ranked lists by reciprocal rank.

    Score-based fusion would need the two scorers to be on a comparable scale,
    and BM25 and cosine similarity are not - BM25 is unbounded and corpus
    dependent, cosine is bounded. Rank is the only thing they share.
    """
    fused: dict[int, float] = {}
    for indices, weight in zip(ranked_indices, weights):
        for rank, index in enumerate(indices):
            fused[int(index)] = fused.get(int(index), 0.0) + weight / (RRF_K + rank + 1)
    return fused


def _spread(fused: dict[int, float]) -> dict[int, float]:
    """Rescale fused scores across the candidate set to span [0, 1].

    Raw RRF scores live in a very narrow band -- with one list, rank 1 scores
    1/61 and rank 40 scores 1/100. A 1.3x metadata boost applied to numbers
    that close swamps the relevance signal entirely, and the reranker starts
    returning whatever the client happens to hold regardless of the question.
    Spreading the band first means a multiplier adjusts the ranking instead of
    replacing it.
    """
    if not fused:
        return {}
    values = list(fused.values())
    low, high = min(values), max(values)
    if high - low < 1e-12:
        return {index: 1.0 for index in fused}
    return {index: (score - low) / (high - low) for index, score in fused.items()}


def _diversify(ranked: list[Chunk], k: int, max_per_doc: int) -> list[Chunk]:
    """Take the top ``k``, allowing at most ``max_per_doc`` from one document.

    Without this, a long document that matches well fills the entire context
    window with adjacent chunks saying nearly the same thing, and the second
    source that would have corroborated or contradicted it never appears.
    """
    taken: dict[str, int] = {}
    out: list[Chunk] = []
    for chunk in ranked:
        if taken.get(chunk.meta.doc_id, 0) >= max_per_doc:
            continue
        taken[chunk.meta.doc_id] = taken.get(chunk.meta.doc_id, 0) + 1
        out.append(chunk)
        if len(out) == k:
            break
    return out


class Retriever:
    """Search over the chunked knowledge base."""

    def __init__(
        self,
        chunks: tuple[Chunk, ...] | None = None,
        embedder: DatabricksEmbedder | None = None,
    ):
        self.chunks = chunks if chunks is not None else build_chunks()
        self.bm25 = BM25Index(self.chunks)
        self.embedder = embedder
        self.matrix: np.ndarray | None = None

    @property
    def semantic_available(self) -> bool:
        return self.matrix is not None

    @classmethod
    async def build(cls, chunks: tuple[Chunk, ...] | None = None) -> "Retriever":
        """Construct and, if an embedding endpoint answers, index densely."""
        retriever = cls(chunks, embedder=await resolve_embedder())
        await retriever.index()
        return retriever

    async def index(self) -> None:
        """Embed the corpus. A failure here downgrades to keyword-only."""
        if self.embedder is None:
            log.warning("no embedder; retrieval is keyword-only")
            return
        try:
            self.matrix = await self.embedder.embed(
                [f"{c.meta.title}\n{c.text}" for c in self.chunks]
            )
            log.info("indexed %d chunks at %d dimensions", *self.matrix.shape)
        except Exception:
            # An endpoint that answered the probe can still fail on a thousand
            # chunks -- quota, payload size, a cold model. Keyword search is a
            # worse product; a chat that will not start is no product.
            log.exception("dense index build failed; falling back to keyword-only")
            self.matrix = None

    # --- access control ----------------------------------------------------

    def _visible(self, index: int, client: Client | None, filters: Filters) -> bool:
        """Whether chunk ``index`` may be returned at all.

        Client scoping is a hard filter and never a ranking signal. A boost can
        be outweighed; a filter cannot, and returning another client's meeting
        note is the one failure here that is not recoverable by apologising.
        """
        meta = self.chunks[index].meta

        if meta.client_id is not None and (
            client is None or meta.client_id != client.client_id
        ):
            return False
        if not filters.include_superseded and meta.superseded_by:
            return False
        if filters.doc_types and meta.doc_type not in filters.doc_types:
            return False
        if filters.languages and meta.language not in filters.languages:
            return False
        if filters.authorities and meta.authority not in filters.authorities:
            return False
        return True

    # --- search ------------------------------------------------------------

    def _top(self, scores: np.ndarray, allowed: np.ndarray, depth: int) -> np.ndarray:
        """Indices of the highest-scoring allowed rows, best first."""
        masked = np.where(allowed, scores, -np.inf)
        depth = min(depth, int(allowed.sum()))
        if depth <= 0:
            return np.empty(0, dtype=np.int64)
        top = np.argpartition(-masked, depth - 1)[:depth]
        return top[np.argsort(-masked[top])]

    async def search(
        self,
        query: str,
        *,
        client: Client | None = None,
        k: int = 6,
        filters: Filters | None = None,
        boosts: dict[str, float] | None = None,
        today: date | None = None,
        max_per_doc: int = 2,
    ) -> list[Chunk]:
        """Top ``k`` chunks for ``query``, hybrid-fused and reranked."""
        filters = filters or Filters()
        allowed = np.array(
            [self._visible(i, client, filters) for i in range(len(self.chunks))]
        )
        if not allowed.any():
            return []

        keyword_ranked = self._top(
            self.bm25.scores(query, boosts=boosts), allowed, CANDIDATE_DEPTH
        )
        ranked_lists = [keyword_ranked]
        # Weighted toward the dense list: BM25 wins on exact scheme names, which
        # the domain-term boosts already amplify, so leaving them equal
        # double-counts the lexical signal.
        weights = [1.0]

        if self.matrix is not None and self.embedder is not None:
            vector = await self.embedder.embed_query(query)
            ranked_lists.append(
                self._top(self.matrix @ vector, allowed, CANDIDATE_DEPTH)
            )
            weights.append(1.2)

        fused = _rrf(ranked_lists, weights)
        candidates = [
            self.chunks[i].with_scores(fused=score)
            for i, score in _spread(fused).items()
        ]
        ranked = rerank(
            candidates,
            client=client,
            today=today,
            penalise_superseded=not filters.include_superseded,
        )
        return _diversify(ranked, k, max_per_doc)


def format_context(chunks: list[Chunk]) -> str:
    """Render retrieved chunks for a prompt, with citable source labels."""
    if not chunks:
        return "No sources retrieved."
    return "\n\n".join(
        f"[S{i}] {chunk.meta.title} "
        f"({chunk.meta.authority}"
        f"{', ' + chunk.meta.effective_date.isoformat() if chunk.meta.effective_date else ''})\n"
        f"{chunk.text}"
        for i, chunk in enumerate(chunks, start=1)
    )
