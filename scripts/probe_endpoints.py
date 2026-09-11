"""Report which serving endpoints this workspace exposes, and which embed.

Run inside Databricks (notebook cell or web terminal, from the repo root):

    python scripts/probe_endpoints.py

Nothing else in the app depends on this — it exists so the retrieval layer can
be configured from fact rather than from guesswork about endpoint naming.
"""

import asyncio
import sys

import httpx

from app.config import EMBED_CANDIDATES, databricks_config
from app.rag.embeddings import DatabricksEmbedder


def list_endpoints() -> list[dict]:
    cfg = databricks_config()
    response = httpx.get(
        f"{cfg.host.rstrip('/')}/api/2.0/serving-endpoints",
        headers=cfg.authenticate(),
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json().get("endpoints", [])


async def try_embedding(model: str) -> str:
    """Call ``model`` once and describe what came back."""
    try:
        vectors = await DatabricksEmbedder(model).embed(["hello"])
    except Exception as exc:
        return f"FAIL  {type(exc).__name__}: {str(exc)[:160]}"
    return f"OK    {vectors.shape[1]} dimensions"


async def main() -> int:
    try:
        endpoints = list_endpoints()
    except Exception as exc:
        print(f"Could not list serving endpoints: {exc}")
        return 1

    print(f"{len(endpoints)} serving endpoint(s) visible to this principal:\n")
    for endpoint in sorted(endpoints, key=lambda e: e.get("name", "")):
        task = endpoint.get("task") or "-"
        state = (endpoint.get("state") or {}).get("ready", "-")
        print(f"  {endpoint.get('name', '?'):<40} task={task:<24} ready={state}")

    # Anything the workspace itself labels as embeddings, plus our defaults, in
    # case the task field is absent (it is, on some workspace versions).
    named = [e.get("name", "") for e in endpoints if "embedding" in (e.get("task") or "")]
    to_try = list(dict.fromkeys(named + list(EMBED_CANDIDATES)))

    print("\nEmbedding probe:\n")
    for model in to_try:
        print(f"  {model:<40} {await try_embedding(model)}")

    print(
        "\nSet AURA_EMBED_MODEL in app.yaml to whichever reported OK. "
        "If none did, retrieval falls back to keyword-only search."
    )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
