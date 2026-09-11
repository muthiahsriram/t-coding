"""Working out which client the app is talking to.

Two sources, in priority order:

1. The chat-profile picker in the UI, if the user chose one.
2. The SSO identity Databricks Apps injects after the workspace login.

The picker wins deliberately. Access to this app is restricted to a handful of
corporate IDs, so SSO can only ever resolve to one person — useless for showing
that different clients see different portfolios. SSO establishes *who is
operating* the app; the picker chooses *whose portfolio* is on screen.

Security note: ``x-forwarded-*`` headers are only trustworthy because the
Databricks Apps proxy sets them and strips any inbound copy. If this app were
ever reachable directly, a caller could set the header themselves and become
any client. Identity must never be taken from a header on an unproxied route.
"""

import os

from app.domain import Client
from app.seed import BOOK, DEFAULT_CLIENT_ID

# Databricks Apps forward the authenticated workspace user in these, most
# specific first.
IDENTITY_HEADERS = (
    "x-forwarded-email",
    "x-forwarded-preferred-username",
    "x-forwarded-user",
)


def normalise_headers(environ: dict | None) -> dict[str, str]:
    """Flatten a WSGI environ into lowercase dashed header names.

    Socket.IO hands us WSGI-style keys (``HTTP_X_FORWARDED_EMAIL``), but plain
    ASGI header dicts (``x-forwarded-email``) show up too depending on the
    transport. Accepting both means the caller never has to care which.
    """
    headers: dict[str, str] = {}
    for key, value in (environ or {}).items():
        if not isinstance(key, str) or not isinstance(value, str):
            continue
        name = key.lower()
        if name.startswith("http_"):
            name = name[len("http_") :]
        headers[name.replace("_", "-")] = value
    return headers


def signed_in_email(environ: dict | None) -> str | None:
    """The workspace user operating the app, if SSO gave us one."""
    headers = normalise_headers(environ)
    for header in IDENTITY_HEADERS:
        value = headers.get(header, "").strip()
        if value:
            return value.lower()
    return None


def _load_client_map() -> dict[str, str]:
    """Map email -> client id.

    Seeded clients map to themselves. ``AURA_CLIENT_MAP`` adds real workspace
    users on top, so a corporate ID can be pointed at a demo client without a
    code change:

        AURA_CLIENT_MAP="jane.doe@fedex.com:C002,john.roe@fedex.com:C005"
    """
    mapping = {client.email.lower(): cid for cid, client in BOOK.items()}

    for pair in os.environ.get("AURA_CLIENT_MAP", "").split(","):
        pair = pair.strip()
        if not pair:
            continue
        if ":" not in pair:
            raise ValueError(f"AURA_CLIENT_MAP entry is not email:client_id -> {pair!r}")
        email, client_id = (part.strip() for part in pair.split(":", 1))
        if client_id not in BOOK:
            # Fail at startup rather than silently serving the wrong portfolio.
            raise ValueError(f"AURA_CLIENT_MAP points at unknown client {client_id!r}")
        mapping[email.lower()] = client_id

    return mapping


CLIENT_BY_EMAIL = _load_client_map()


def profile_label(client: Client) -> str:
    """Display name for the chat-profile picker."""
    return f"{client.name} ({client.client_id})"


CLIENT_BY_PROFILE = {profile_label(client): cid for cid, client in BOOK.items()}


def resolve_client_id(
    environ: dict | None = None, profile: str | None = None
) -> str:
    """Pick the client whose portfolio should be on screen."""
    if profile and profile in CLIENT_BY_PROFILE:
        return CLIENT_BY_PROFILE[profile]

    email = signed_in_email(environ)
    if email and email in CLIENT_BY_EMAIL:
        return CLIENT_BY_EMAIL[email]

    return DEFAULT_CLIENT_ID
