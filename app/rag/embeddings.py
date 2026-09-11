"""Text embeddings served by Databricks Foundation Model APIs.

The workspace is the constraint here. Databricks serves embedding models under
names that differ between workspaces, and some workspaces serve none at all, so
this module *probes* rather than hardcoding an endpoint and discovering the
mistake at query time. If nothing answers, ``resolve_embedder`` returns ``None``
and the retrieval layer above degrades to keyword-only search instead of
failing — a hybrid index missing its dense half still returns results.

Vectors are L2-normalised on the way out. Cosine similarity then reduces to a
dot product, which means the whole corpus search is one ``matrix @ vector`` and
we need no vector database for a corpus this size.
"""

from __future__ import annotations

import asyncio
import logging

import numpy as np

from app.config import EMBED_BATCH, EMBED_CANDIDATES, EMBED_MODEL
from app.llm import get_client

log = logging.getLogger(__name__)

# BGE models were trained with an asymmetric instruction: queries carry a
# prefix, documents do not. Omitting it costs a few points of recall. GTE was
# trained symmetrically and needs nothing.
_QUERY_PREFIX = {
    "databricks-bge-large-en": "Represent this sentence for searching relevant passages: ",
}


class EmbeddingUnavailable(RuntimeError):
    """No embedding endpoint in this workspace answered."""


def _normalise(vectors: np.ndarray) -> np.ndarray:
    """Scale each row to unit length, leaving zero rows alone."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1.0, norms)


# A probe must fail fast. The shared chat client is configured for generation:
# a 180s timeout with three retries, which is right for a long completion and
# catastrophic for a liveness check -- two dead candidates would stall the
# first message for twenty minutes behind what is meant to be a quick question.
PROBE_TIMEOUT = 15.0


class DatabricksEmbedder:
    """Embeds text through one named Databricks serving endpoint."""

    def __init__(
        self,
        model: str,
        *,
        batch_size: int = EMBED_BATCH,
        timeout: float | None = None,
        max_retries: int | None = None,
    ):
        self.model = model
        self.batch_size = batch_size
        self.timeout = timeout
        self.max_retries = max_retries
        self._dimensions: int | None = None

    @property
    def dimensions(self) -> int | None:
        """Vector width, known only once the endpoint has answered."""
        return self._dimensions

    async def embed(self, texts: list[str], *, kind: str = "document") -> np.ndarray:
        """Embed ``texts`` into a normalised ``(len(texts), d)`` float32 array.

        ``kind`` is ``"query"`` or ``"document"``; it selects the instruction
        prefix for models that were trained asymmetrically.
        """
        if not texts:
            return np.zeros((0, self._dimensions or 0), dtype=np.float32)

        prefix = _QUERY_PREFIX.get(self.model, "") if kind == "query" else ""
        payload = [prefix + text for text in texts]

        client = get_client()
        if self.max_retries is not None:
            client = client.with_options(max_retries=self.max_retries)
        options = {"timeout": self.timeout} if self.timeout is not None else {}

        batches = [
            payload[i : i + self.batch_size]
            for i in range(0, len(payload), self.batch_size)
        ]

        # Sequential, not gathered: a few hundred concurrent requests at index
        # build time is how you get rate-limited off your own endpoint.
        rows: list[list[float]] = []
        for batch in batches:
            response = await client.embeddings.create(
                model=self.model, input=batch, **options
            )
            # The endpoint is not contractually obliged to preserve input order.
            for item in sorted(response.data, key=lambda d: d.index):
                rows.append(item.embedding)

        vectors = np.asarray(rows, dtype=np.float32)
        self._dimensions = int(vectors.shape[1])
        return _normalise(vectors)

    async def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query into a 1-D vector."""
        return (await self.embed([text], kind="query"))[0]


async def probe(model: str, *, timeout: float | None = None) -> bool:
    """True if ``model`` is a live embedding endpoint we may call.

    Bounded twice over: once through the request timeout, and once with
    ``wait_for`` in case the hang is somewhere the HTTP timeout does not reach
    -- credential resolution on this workspace has blocked indefinitely before.
    """
    # Read at call time, not bound as a default: a module-level default is
    # captured at import and silently ignores anything that tunes the constant
    # afterwards, which is exactly the kind of knob that looks like it works.
    timeout = PROBE_TIMEOUT if timeout is None else timeout
    embedder = DatabricksEmbedder(model, timeout=timeout, max_retries=0)
    try:
        await asyncio.wait_for(embedder.embed(["probe"]), timeout=timeout)
    except asyncio.TimeoutError:
        log.info("embedding endpoint %s did not answer within %.0fs", model, timeout)
        return False
    except Exception as exc:  # endpoint missing, no permission, wrong task type
        log.info("embedding endpoint %s unavailable: %s", model, exc)
        return False
    return True


async def resolve_embedder(
    candidates: tuple[str, ...] = EMBED_CANDIDATES,
) -> DatabricksEmbedder | None:
    """First candidate endpoint that answers, or ``None`` if none do.

    An explicit ``AURA_EMBED_MODEL`` is trusted without probing: if someone
    named an endpoint and got it wrong, a loud failure beats silently falling
    back to keyword search and wondering why recall collapsed.
    """
    if EMBED_MODEL:
        return DatabricksEmbedder(EMBED_MODEL)

    for model in candidates:
        if await probe(model):
            log.info("using embedding endpoint %s", model)
            return DatabricksEmbedder(model)

    log.warning(
        "no embedding endpoint available (tried %s); "
        "semantic search is disabled and retrieval will use keyword search only",
        ", ".join(candidates),
    )
    return None


def resolve_embedder_sync(
    candidates: tuple[str, ...] = EMBED_CANDIDATES,
) -> DatabricksEmbedder | None:
    """Blocking ``resolve_embedder``, for index-build scripts and notebooks."""
    return asyncio.run(resolve_embedder(candidates))
