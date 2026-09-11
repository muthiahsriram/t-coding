import asyncio
from types import SimpleNamespace

import httpx
import pytest

from app import config, llm


def chunk(content):
    """Build a streaming chunk shaped like the serving endpoint's response."""
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=content))])


KEEPALIVE = SimpleNamespace(choices=[])


class FakeStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        async def gen():
            for c in self._chunks:
                yield c

        return gen()


def fake_client(chunks, calls=None):
    async def create(**kwargs):
        if calls is not None:
            calls.append(kwargs)
        return FakeStream(chunks)

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


async def collect(messages):
    return [delta async for delta in llm.stream_chat(messages)]


class RotatingConfig:
    """Stands in for databricks-sdk Config, minting a new token each call."""

    def __init__(self, host="https://example.cloud.databricks.com"):
        self.host = host
        self.calls = 0

    def authenticate(self):
        self.calls += 1
        return {"Authorization": f"Bearer token-{self.calls}"}


# --- credential resolution -------------------------------------------------


def test_serving_url_is_built_from_the_workspace_host_without_a_double_slash():
    assert (
        config.serving_base_url(RotatingConfig("https://example.cloud.databricks.com/"))
        == "https://example.cloud.databricks.com/serving-endpoints"
    )


def test_a_personal_token_bypasses_the_sdk_credential_chain(monkeypatch):
    """Config()'s chain probes cloud metadata and can hang; PATs must not."""
    monkeypatch.setenv("DATABRICKS_HOST", "https://example.cloud.databricks.com/")
    monkeypatch.setenv("DATABRICKS_TOKEN", "dapi-secret")
    monkeypatch.setattr(
        config, "Config", lambda: pytest.fail("Config() must not be constructed")
    )

    cfg = config.databricks_config()

    assert cfg.host == "https://example.cloud.databricks.com"
    assert cfg.authenticate() == {"Authorization": "Bearer dapi-secret"}


def test_a_host_without_a_token_falls_through_to_the_sdk(monkeypatch):
    """Half a PAT is not a PAT — don't build a client that 401s on every call."""
    monkeypatch.setenv("DATABRICKS_HOST", "https://example.cloud.databricks.com")
    monkeypatch.delenv("DATABRICKS_TOKEN", raising=False)
    monkeypatch.setattr(config, "Config", lambda: SimpleNamespace(host=None))

    with pytest.raises(RuntimeError, match="DATABRICKS_CLIENT_ID"):
        config.databricks_config()


def test_credential_chain_failures_surface_as_a_readable_error(monkeypatch):
    """databricks-sdk raises its own error types; they must not leak upward."""
    monkeypatch.delenv("DATABRICKS_HOST", raising=False)
    monkeypatch.delenv("DATABRICKS_TOKEN", raising=False)

    def explode():
        raise ValueError("default auth: cannot configure default credentials")

    monkeypatch.setattr(config, "Config", explode)

    with pytest.raises(RuntimeError, match="No Databricks credentials found"):
        config.databricks_config()


# --- OAuth token refresh ---------------------------------------------------


def test_auth_hook_mints_a_new_token_for_every_request():
    """App service-principal tokens expire; a captured one dies after an hour."""
    auth = llm.DatabricksAuth(RotatingConfig())
    url = "https://example.cloud.databricks.com/serving-endpoints/m/invocations"

    first = httpx.Request("POST", url)
    next(auth.auth_flow(first))
    second = httpx.Request("POST", url)
    next(auth.auth_flow(second))

    assert first.headers["authorization"] == "Bearer token-1"
    assert second.headers["authorization"] == "Bearer token-2"


def test_auth_hook_overrides_the_openai_sdk_placeholder_api_key():
    """The SDK always sets Authorization from api_key; ours must win."""
    auth = llm.DatabricksAuth(RotatingConfig())
    request = httpx.Request(
        "POST",
        "https://example.cloud.databricks.com/serving-endpoints/m/invocations",
        headers={"Authorization": "Bearer databricks-oauth"},
    )

    next(auth.auth_flow(request))

    assert request.headers["authorization"] == "Bearer token-1"


# --- streaming -------------------------------------------------------------


def test_stream_chat_yields_text_and_skips_keepalives_and_empty_deltas(monkeypatch):
    """Keepalives carry no choices; the final chunk carries no content."""
    monkeypatch.setattr(
        llm,
        "get_client",
        lambda: fake_client([chunk("Your "), KEEPALIVE, chunk("net worth"), chunk(None)]),
    )

    assert asyncio.run(collect([{"role": "user", "content": "hi"}])) == [
        "Your ",
        "net worth",
    ]


def test_stream_chat_forwards_the_full_history_and_requests_streaming(monkeypatch):
    """Dropping history here is how multi-turn silently regresses to one-shot."""
    calls = []
    monkeypatch.setattr(llm, "get_client", lambda: fake_client([chunk("ok")], calls))

    history = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "second"},
    ]
    asyncio.run(collect(history))

    assert calls[0]["messages"] == history
    assert calls[0]["stream"] is True
    assert calls[0]["model"] == config.CHAT_MODEL
