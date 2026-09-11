import numpy as np
import pytest

from app.rag import embeddings


class _FakeEmbeddings:
    """Stands in for ``client.embeddings``, recording what it was asked."""

    def __init__(self, dimensions=4, fail_for=(), shuffle=False):
        self.dimensions = dimensions
        self.fail_for = fail_for
        self.shuffle = shuffle
        self.calls: list[tuple[str, list[str]]] = []

    async def create(self, *, model, input):
        self.calls.append((model, list(input)))
        if model in self.fail_for:
            raise RuntimeError(f"endpoint {model} does not exist")

        data = [
            type("Item", (), {"index": i, "embedding": [float(i + 1)] * self.dimensions})
            for i in range(len(input))
        ]
        if self.shuffle:
            data = list(reversed(data))
        return type("Response", (), {"data": data})


@pytest.fixture
def fake_client(monkeypatch):
    def install(**kwargs):
        stub = _FakeEmbeddings(**kwargs)
        client = type("Client", (), {"embeddings": stub})
        monkeypatch.setattr(embeddings, "get_client", lambda: client)
        return stub

    return install


@pytest.mark.asyncio
async def test_vectors_come_back_unit_length(fake_client):
    """Cosine similarity is only a dot product if the vectors are normalised."""
    fake_client()

    vectors = await embeddings.DatabricksEmbedder("m").embed(["a", "b"])

    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    assert vectors.dtype == np.float32


@pytest.mark.asyncio
async def test_a_zero_vector_does_not_divide_by_zero(fake_client):
    fake_client(dimensions=3)
    zeros = embeddings._normalise(np.zeros((1, 3), dtype=np.float32))

    assert np.all(np.isfinite(zeros))


@pytest.mark.asyncio
async def test_long_input_is_split_into_batches(fake_client):
    """Endpoints reject oversized input arrays, so the split must actually happen."""
    stub = fake_client()

    vectors = await embeddings.DatabricksEmbedder("m", batch_size=2).embed(
        [f"chunk {i}" for i in range(5)]
    )

    assert [len(sent) for _, sent in stub.calls] == [2, 2, 1]
    assert vectors.shape == (5, 4)


@pytest.mark.asyncio
async def test_results_are_reordered_to_match_the_input(fake_client):
    """The endpoint is not obliged to return items in input order."""
    fake_client(shuffle=True)

    vectors = await embeddings.DatabricksEmbedder("m").embed(["a", "b", "c"])

    # Row i was built from value i+1, so a correct reorder is strictly ascending.
    assert list(np.argsort(vectors[:, 0])) == [0, 1, 2]


@pytest.mark.asyncio
async def test_embedding_nothing_costs_no_request(fake_client):
    """Index builds hit empty batches on filtered corpora; that must be free."""
    stub = fake_client()

    assert (await embeddings.DatabricksEmbedder("m").embed([])).shape[0] == 0
    assert stub.calls == []


@pytest.mark.asyncio
async def test_bge_gets_its_query_prefix_and_gte_does_not(fake_client):
    """BGE was trained asymmetrically; dropping the prefix silently costs recall."""
    stub = fake_client()

    await embeddings.DatabricksEmbedder("databricks-bge-large-en").embed_query("cpf")
    await embeddings.DatabricksEmbedder("databricks-gte-large-en").embed_query("cpf")

    assert stub.calls[0][1] == [
        "Represent this sentence for searching relevant passages: cpf"
    ]
    assert stub.calls[1][1] == ["cpf"]


@pytest.mark.asyncio
async def test_documents_never_get_the_query_prefix(fake_client):
    stub = fake_client()

    await embeddings.DatabricksEmbedder("databricks-bge-large-en").embed(["cpf"])

    assert stub.calls[0][1] == ["cpf"]


# --- endpoint resolution ---------------------------------------------------


@pytest.mark.asyncio
async def test_resolution_skips_endpoints_the_workspace_does_not_serve(
    fake_client, monkeypatch
):
    monkeypatch.setattr(embeddings, "EMBED_MODEL", "")
    fake_client(fail_for=("databricks-gte-large-en",))

    embedder = await embeddings.resolve_embedder(
        ("databricks-gte-large-en", "databricks-bge-large-en")
    )

    assert embedder is not None
    assert embedder.model == "databricks-bge-large-en"


@pytest.mark.asyncio
async def test_no_embedding_endpoint_degrades_instead_of_raising(
    fake_client, monkeypatch
):
    """Keyword-only retrieval is a worse product; a crashed app is no product."""
    monkeypatch.setattr(embeddings, "EMBED_MODEL", "")
    fake_client(fail_for=("a", "b"))

    assert await embeddings.resolve_embedder(("a", "b")) is None


@pytest.mark.asyncio
async def test_an_explicitly_configured_endpoint_is_not_probed(
    fake_client, monkeypatch
):
    """A named endpoint that is wrong should fail loudly, not fall back quietly."""
    monkeypatch.setattr(embeddings, "EMBED_MODEL", "my-endpoint")
    stub = fake_client()

    embedder = await embeddings.resolve_embedder()

    assert embedder is not None and embedder.model == "my-endpoint"
    assert stub.calls == []
