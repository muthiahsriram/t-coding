from datetime import date

import numpy as np
import pytest

from app.rag import keyword
from app.rag.corpus import _parse_front_matter, build_chunks, build_corpus
from app.rag.documents import DocMeta, Document, chunk_document
from app.rag.retriever import Filters, Retriever
from app.seed import get_client

TODAY = date(2026, 1, 1)


# --- corpus ----------------------------------------------------------------


def test_the_corpus_clears_the_required_scale():
    """The brief asks for semantic and keyword search over 1000+ chunks."""
    assert len(build_corpus()) > 10
    assert len(build_chunks()) >= 1000


def test_the_corpus_is_identical_across_builds():
    """Retrieval metrics are only comparable across runs on a fixed corpus."""
    first = [c.chunk_id for c in build_chunks()]
    build_chunks.cache_clear()
    assert [c.chunk_id for c in build_chunks()] == first


def test_the_corpus_covers_four_languages():
    languages = {d.meta.language for d in build_corpus()}
    assert {"en", "zh", "ms", "ta"} <= languages


def test_unknown_front_matter_is_rejected_rather_than_ignored():
    """A typo'd key would silently drop metadata the reranker depends on."""
    with pytest.raises(ValueError, match="autority"):
        _parse_front_matter("---\ntitle: X\nautority: MAS\n---\nbody", "doc")


def test_a_document_without_front_matter_is_rejected():
    with pytest.raises(ValueError, match="front matter"):
        _parse_front_matter("just some text", "doc")


# --- chunking --------------------------------------------------------------


def _document(text: str) -> Document:
    return Document(meta=DocMeta(doc_id="d", title="T", doc_type="policy"), text=text)


def test_chunks_overlap_so_a_rule_split_at_the_boundary_stays_findable():
    paragraphs = "\n\n".join(" ".join(f"w{i}-{j}" for j in range(80)) for i in range(4))
    chunks = chunk_document(_document(paragraphs), max_words=100, overlap=20)

    assert len(chunks) > 1
    tail = chunks[0].text.split()[-20:]
    assert tail == chunks[1].text.split()[:20]


def test_a_paragraph_longer_than_the_chunk_size_is_windowed_not_dropped():
    long_paragraph = " ".join(f"w{i}" for i in range(500))
    chunks = chunk_document(_document(long_paragraph), max_words=100, overlap=20)

    assert len(chunks) >= 5
    assert "w499" in chunks[-1].text


def test_chunk_ids_are_stable_and_unique():
    ids = [c.chunk_id for c in build_chunks()]
    assert len(set(ids)) == len(ids)


# --- keyword search --------------------------------------------------------


def test_cjk_is_bigrammed_so_chinese_documents_are_not_treated_as_tiny():
    """The Latin-only tokenizer made zh docs look ~3 tokens long, and BM25
    length normalisation then ranked them above everything on any Latin term."""
    tokens = keyword.tokenize("我可以往SRS存多少钱")

    assert "srs" in tokens  # the acronym survives intact
    assert any(len(t) == 2 and not t.isascii() for t in tokens)  # CJK bigrams


def test_an_acronym_glued_to_cjk_is_not_shredded():
    assert "cpfis" in keyword.tokenize("我的CPFIS账户")


def test_stopwords_are_dropped():
    assert keyword.tokenize("what is the cap") == ["cap"]


def test_domain_terms_outweigh_ordinary_ones():
    index = keyword.BM25Index(build_chunks())

    boosted = index.scores("ABSD")
    plain = index.scores("ABSD", boosts={"absd": 1.0})

    assert boosted.max() > plain.max()


def test_a_query_of_pure_stopwords_scores_nothing_rather_than_raising():
    index = keyword.BM25Index(build_chunks())
    assert index.scores("what is the").max() == 0.0


# --- retrieval -------------------------------------------------------------


@pytest.fixture(scope="module")
def retriever() -> Retriever:
    """Keyword-only: the path taken when the workspace serves no embeddings."""
    return Retriever()


async def test_keyword_only_retrieval_still_answers(retriever):
    """No embedding endpoint must degrade the product, not break it."""
    assert not retriever.semantic_available

    results = await retriever.search("ABSD on a second property", k=3, today=TODAY)

    assert results
    assert results[0].meta.doc_id == "property-stamp-duties"


async def test_a_withdrawn_document_is_excluded_by_default(retriever):
    """Textually perfect and operationally wrong is the dangerous combination."""
    query = "keep Special Account savings after 55 by investing them"

    results = await retriever.search(query, k=6, today=TODAY)

    assert all(c.meta.superseded_by is None for c in results)
    assert any(c.meta.doc_id == "cpf-special-account-closure" for c in results)


async def test_a_withdrawn_document_is_still_reachable_when_asked_for(retriever):
    """A client may ask why the old advice changed; the old text has to exist."""
    results = await retriever.search(
        "withdrawn guidance on retaining Special Account savings",
        k=8,
        filters=Filters(include_superseded=True),
        today=TODAY,
    )

    assert any(c.meta.doc_id == "cpf-shielding-legacy" for c in results)


async def test_one_client_never_sees_another_clients_meeting_notes(retriever):
    """Client scoping is a filter, not a boost: a boost can be outweighed."""
    c002 = get_client("C002")

    results = await retriever.search(
        "meeting note review of the mandate", client=c002, k=10, today=TODAY
    )

    notes = [c for c in results if c.meta.doc_type == "meeting_note"]
    assert notes, "expected the client's own notes to be reachable"
    assert all(c.meta.client_id == "C002" for c in notes)


async def test_meeting_notes_are_invisible_with_no_client_in_context(retriever):
    results = await retriever.search(
        "meeting note review of the mandate", client=None, k=10, today=TODAY
    )

    assert all(c.meta.client_id is None for c in results)


async def test_results_are_not_all_from_one_document(retriever):
    """Six adjacent chunks of one document is a wasted context window."""
    results = await retriever.search("CPF interest rates and retirement sums", k=6, today=TODAY)

    counts: dict[str, int] = {}
    for chunk in results:
        counts[chunk.meta.doc_id] = counts.get(chunk.meta.doc_id, 0) + 1
    assert max(counts.values()) <= 2


async def test_metadata_filters_narrow_the_candidate_set(retriever):
    results = await retriever.search(
        "concentration limits", k=5, filters=Filters(doc_types=("policy",)), today=TODAY
    )

    assert results
    assert all(c.meta.doc_type == "policy" for c in results)


async def test_a_filter_that_matches_nothing_returns_nothing(retriever):
    assert await retriever.search("anything", filters=Filters(languages=("fr",))) == []


# --- the dense half --------------------------------------------------------


class _StubEmbedder:
    """Deterministic hashed embeddings: no workspace, but a real dense path."""

    def __init__(self, dimensions: int = 32):
        self.dimensions = dimensions

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        for token in keyword.tokenize(text):
            vector[hash(token) % self.dimensions] += 1.0
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    async def embed(self, texts, *, kind="document"):
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)
        return np.vstack([self._vector(t) for t in texts])

    async def embed_query(self, text):
        return self._vector(text)


async def test_the_dense_index_is_used_when_an_embedder_is_present():
    chunks = build_chunks()[:200]
    retriever = Retriever(chunks, embedder=_StubEmbedder())
    await retriever.index()

    assert retriever.semantic_available
    assert retriever.matrix.shape == (200, 32)
    assert await retriever.search("CPF interest", k=3, today=TODAY)


async def test_an_embedder_that_fails_mid_build_falls_back_to_keyword_only():
    """A probe can succeed and the full corpus still blow a quota."""

    class Exploding(_StubEmbedder):
        async def embed(self, texts, *, kind="document"):
            raise RuntimeError("payload too large")

    retriever = Retriever(build_chunks()[:50], embedder=Exploding())
    await retriever.index()

    assert not retriever.semantic_available
    assert await retriever.search("CPF interest", k=3, today=TODAY)
