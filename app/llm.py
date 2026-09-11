"""Async client for Claude served through Databricks Foundation Model APIs.

Databricks exposes its serving endpoints with an OpenAI-compatible schema, so
the official ``openai`` SDK talks to them directly once ``base_url`` points at
``<workspace>/serving-endpoints``. Using the SDK rather than raw ``requests``
buys us streaming delta accumulation, tool-call reassembly across chunks and
429/5xx retries that we would otherwise hand-roll.
"""

from collections.abc import AsyncIterator, Generator

import httpx
from databricks.sdk.core import Config
from openai import AsyncOpenAI

from app.config import (
    CHAT_MODEL,
    MAX_TOKENS,
    TEMPERATURE,
    databricks_config,
    serving_base_url,
)

_client: AsyncOpenAI | None = None


class DatabricksAuth(httpx.Auth):
    """Stamp a freshly-minted Databricks token on every outbound request.

    In a Databricks App we authenticate as an OAuth service principal, and
    those tokens expire within the hour. Baking one into ``api_key`` at client
    construction would work until the app had been running long enough to
    matter, then start returning 401s. ``Config.authenticate()`` returns the
    current token and refreshes it when needed, so doing this per-request is
    what keeps a long-lived app alive.
    """

    def __init__(self, cfg: Config):
        self._cfg = cfg

    def auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, None, None]:
        request.headers.update(self._cfg.authenticate())
        yield request


def get_client() -> AsyncOpenAI:
    """Lazily build the shared client.

    Lazy so that importing this module never requires credentials — tests and
    tooling can import it without a live workspace.
    """
    global _client
    if _client is None:
        cfg = databricks_config()
        _client = AsyncOpenAI(
            # Overwritten per-request by DatabricksAuth; the SDK only requires
            # that it be non-empty.
            api_key="databricks-oauth",
            base_url=serving_base_url(cfg),
            http_client=httpx.AsyncClient(auth=DatabricksAuth(cfg), timeout=180.0),
            max_retries=3,
        )
    return _client


async def stream_chat(
    messages: list[dict], *, model: str = CHAT_MODEL
) -> AsyncIterator[str]:
    """Yield response text deltas for ``messages`` as they arrive."""
    stream = await get_client().chat.completions.create(
        model=model,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        stream=True,
    )
    async for chunk in stream:
        # Databricks emits keepalive chunks with an empty ``choices`` list, and
        # the final chunk carries a finish_reason with no content.
        if not chunk.choices:
            continue
        content = chunk.choices[0].delta.content
        if content:
            yield content


async def complete(messages: list[dict], *, model: str = CHAT_MODEL) -> str:
    """Non-streaming completion, for agent steps with no user-visible output."""
    response = await get_client().chat.completions.create(
        model=model,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    return response.choices[0].message.content or ""
