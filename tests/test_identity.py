import importlib

import pytest

from app import identity
from app.seed import BOOK, DEFAULT_CLIENT_ID


def test_wsgi_style_headers_are_read():
    """Socket.IO hands us HTTP_X_FORWARDED_EMAIL, not x-forwarded-email."""
    environ = {"HTTP_X_FORWARDED_EMAIL": "Jane.Doe@FedEx.com", "REQUEST_METHOD": "GET"}

    assert identity.signed_in_email(environ) == "jane.doe@fedex.com"


def test_plain_header_dicts_are_read_too():
    assert (
        identity.signed_in_email({"x-forwarded-email": "jane.doe@fedex.com"})
        == "jane.doe@fedex.com"
    )


def test_identity_headers_are_tried_most_specific_first():
    """A username is a weaker identity than an email; email must win."""
    environ = {
        "HTTP_X_FORWARDED_USER": "jdoe",
        "HTTP_X_FORWARDED_EMAIL": "jane.doe@fedex.com",
    }

    assert identity.signed_in_email(environ) == "jane.doe@fedex.com"


def test_no_sso_headers_yields_no_identity():
    """Running locally there is no proxy, and that must not raise."""
    assert identity.signed_in_email({"REQUEST_METHOD": "GET"}) is None
    assert identity.signed_in_email(None) is None
    assert identity.signed_in_email({}) is None


def test_a_blank_forwarded_header_is_not_an_identity():
    """Proxies forward the header empty rather than omitting it."""
    assert identity.signed_in_email({"HTTP_X_FORWARDED_EMAIL": "   "}) is None


# --- resolution order ------------------------------------------------------


def test_the_profile_picker_beats_the_sso_identity():
    """Access is limited to a few corporate IDs, so SSO can only ever be one
    person. The picker is what makes multiple clients demonstrable."""
    environ = {"HTTP_X_FORWARDED_EMAIL": BOOK["C002"].email}
    picked = identity.profile_label(BOOK["C005"])

    assert identity.resolve_client_id(environ, picked) == "C005"


def test_sso_identity_is_used_when_no_profile_is_picked():
    environ = {"HTTP_X_FORWARDED_EMAIL": BOOK["C004"].email.upper()}

    assert identity.resolve_client_id(environ, None) == "C004"


def test_an_unknown_user_falls_back_to_the_default_client():
    """A colleague opening the app must get a working demo, not an error."""
    environ = {"HTTP_X_FORWARDED_EMAIL": "nobody@fedex.com"}

    assert identity.resolve_client_id(environ, None) == DEFAULT_CLIENT_ID
    assert identity.resolve_client_id(None, "not a real profile") == DEFAULT_CLIENT_ID


# --- the deployment-time mapping -------------------------------------------


def test_a_corporate_id_can_be_mapped_to_a_client_without_a_code_change(monkeypatch):
    monkeypatch.setenv("AURA_CLIENT_MAP", "jane.doe@fedex.com:C007")
    reloaded = importlib.reload(identity)

    try:
        environ = {"HTTP_X_FORWARDED_EMAIL": "jane.doe@fedex.com"}
        assert reloaded.resolve_client_id(environ, None) == "C007"
    finally:
        monkeypatch.delenv("AURA_CLIENT_MAP")
        importlib.reload(identity)


def test_a_mapping_to_an_unknown_client_fails_at_startup(monkeypatch):
    """Silently serving the wrong person's portfolio is the worse failure."""
    monkeypatch.setenv("AURA_CLIENT_MAP", "jane.doe@fedex.com:C999")

    try:
        with pytest.raises(ValueError, match="C999"):
            importlib.reload(identity)
    finally:
        monkeypatch.delenv("AURA_CLIENT_MAP")
        importlib.reload(identity)


def test_a_malformed_mapping_entry_fails_at_startup(monkeypatch):
    monkeypatch.setenv("AURA_CLIENT_MAP", "jane.doe@fedex.com")

    try:
        with pytest.raises(ValueError, match="email:client_id"):
            importlib.reload(identity)
    finally:
        monkeypatch.delenv("AURA_CLIENT_MAP")
        importlib.reload(identity)


def test_every_client_has_a_unique_picker_label():
    """Two clients sharing a label would make the picker ambiguous."""
    assert len(identity.CLIENT_BY_PROFILE) == len(BOOK)
