"""Runtime configuration and Databricks credential resolution."""

import os
from dataclasses import dataclass

from databricks.sdk.core import Config
from dotenv import load_dotenv

load_dotenv()

# Chat model served by Databricks Foundation Model APIs.
CHAT_MODEL = os.environ.get("AURA_CHAT_MODEL", "databricks-claude-sonnet-4-5")

# Sampling defaults. Temperature 0 keeps agent behaviour reproducible, which
# matters once we start evaluating workflows against fixed test sets.
TEMPERATURE = float(os.environ.get("AURA_TEMPERATURE", "0.0"))
MAX_TOKENS = int(os.environ.get("AURA_MAX_TOKENS", "2048"))

# Embedding endpoint. Which of these a workspace actually serves varies, so we
# probe in order rather than hardcode one and fail at query time. Setting
# AURA_EMBED_MODEL skips the probe entirely.
EMBED_MODEL = os.environ.get("AURA_EMBED_MODEL", "").strip()
EMBED_CANDIDATES: tuple[str, ...] = (
    "databricks-gte-large-en",
    "databricks-bge-large-en",
)

# Databricks embedding endpoints reject oversized input arrays. 32 is well
# inside every published limit and still amortises the round trip.
EMBED_BATCH = int(os.environ.get("AURA_EMBED_BATCH", "32"))

_CREDENTIAL_HELP = (
    "No Databricks credentials found. In a Databricks App these are injected "
    "automatically (DATABRICKS_CLIENT_ID / DATABRICKS_CLIENT_SECRET); "
    "elsewhere set DATABRICKS_HOST and DATABRICKS_TOKEN — see .env.example."
)


@dataclass(frozen=True)
class PersonalToken:
    """A static personal access token.

    Mirrors the slice of ``databricks.sdk.core.Config`` we depend on — a
    ``host`` and an ``authenticate()`` that returns auth headers — so callers
    can treat both sources identically.
    """

    host: str
    token: str

    def authenticate(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}


def databricks_config() -> PersonalToken | Config:
    """Resolve workspace credentials for whichever environment we're in.

    A personal access token in the environment is used directly. Everything
    else defers to ``Config``, which walks the standard Databricks credential
    chain — in a Databricks App that resolves to the OAuth service principal
    and, crucially, *refreshes* it, which a captured bearer string cannot do.

    The PAT branch deliberately bypasses ``Config``: its credential chain
    probes cloud metadata services, which hangs for minutes on networks that
    blackhole those addresses rather than refusing the connection. A hang here
    is a failed startup health check, so the cheap path stays cheap.
    """
    host = os.environ.get("DATABRICKS_HOST")
    token = os.environ.get("DATABRICKS_TOKEN")
    if host and token:
        return PersonalToken(host=host.rstrip("/"), token=token)

    try:
        cfg = Config()
    except Exception as exc:  # databricks-sdk raises its own error types
        raise RuntimeError(_CREDENTIAL_HELP) from exc

    if not cfg.host:
        raise RuntimeError(_CREDENTIAL_HELP)
    return cfg


def serving_base_url(cfg: PersonalToken | Config | None = None) -> str:
    """OpenAI-compatible base URL for the workspace's serving endpoints."""
    cfg = cfg or databricks_config()
    return f"{cfg.host.rstrip('/')}/serving-endpoints"
