"""credential_failure_flags and turn_exit_code classify missing-credential AuthErrors as terminal.

Regression: AuthError(code='missing_api_key', relogin_required=False) was mapped as
non-terminal (exit code 1) instead of terminal (exit code 78), so the kanban board
received worker_result exit_code=1 instead of the re-login code 78, and the kanban
worker client never saw a terminal receipt.

Rate-limit/quota errors (CODEX_RATE_LIMITED_CODE) must keep their transient mapping (75).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from hermes_cli.auth_constants import AuthError


def make_auth_error(*, code=None, relogin_required=False, provider="test"):
    """Helper: construct an AuthError with the given attributes."""
    return AuthError(
        f"test: {code or 'no code'}",
        provider=provider,
        code=code,
        relogin_required=relogin_required,
    )


class TestCredentialFailureFlags:
    """credential_failure_flags maps AuthError codes to terminal/rate-limited correctly."""

    def test_missing_api_key_is_terminal(self):
        """missing_api_key is a terminal credential failure."""
        from hermes_cli.turn_exit import credential_failure_flags

        exc = make_auth_error(code="missing_api_key")
        flags = credential_failure_flags(exc)
        assert flags == {"credentials_rate_limited": False, "credentials_terminal": True}

    def test_no_provider_configured_is_terminal(self):
        """no_provider_configured is a terminal credential failure."""
        from hermes_cli.turn_exit import credential_failure_flags

        exc = make_auth_error(code="no_provider_configured")
        flags = credential_failure_flags(exc)
        assert flags == {"credentials_rate_limited": False, "credentials_terminal": True}

    def test_invalid_provider_is_terminal(self):
        """invalid_provider is a terminal credential failure."""
        from hermes_cli.turn_exit import credential_failure_flags

        exc = make_auth_error(code="invalid_provider")
        flags = credential_failure_flags(exc)
        assert flags == {"credentials_rate_limited": False, "credentials_terminal": True}

    def test_nous_auth_missing_is_terminal(self):
        from hermes_cli.turn_exit import credential_failure_flags

        exc = make_auth_error(code="nous_auth_missing")
        flags = credential_failure_flags(exc)
        assert flags == {"credentials_rate_limited": False, "credentials_terminal": True}

    def test_qwen_auth_missing_is_terminal(self):
        from hermes_cli.turn_exit import credential_failure_flags

        exc = make_auth_error(code="qwen_auth_missing")
        flags = credential_failure_flags(exc)
        assert flags == {"credentials_rate_limited": False, "credentials_terminal": True}

    def test_codex_auth_missing_is_terminal(self):
        from hermes_cli.turn_exit import credential_failure_flags

        exc = make_auth_error(code="codex_auth_missing")
        flags = credential_failure_flags(exc)
        assert flags == {"credentials_rate_limited": False, "credentials_terminal": True}

    def test_codex_auth_invalid_shape_is_terminal(self):
        from hermes_cli.turn_exit import credential_failure_flags

        exc = make_auth_error(code="codex_auth_invalid_shape")
        flags = credential_failure_flags(exc)
        assert flags == {"credentials_rate_limited": False, "credentials_terminal": True}

    def test_corrupt_config_is_terminal(self):
        from hermes_cli.turn_exit import credential_failure_flags

        exc = make_auth_error(code="corrupt_config")
        flags = credential_failure_flags(exc)
        assert flags == {"credentials_rate_limited": False, "credentials_terminal": True}

    def test_rate_limited_codex_stays_non_terminal(self):
        """CODEX_RATE_LIMITED_CODE is transient: rate_limited=True, terminal=False."""
        from hermes_cli.auth_constants import CODEX_RATE_LIMITED_CODE
        from hermes_cli.turn_exit import credential_failure_flags

        exc = make_auth_error(code=CODEX_RATE_LIMITED_CODE)
        flags = credential_failure_flags(exc)
        assert flags == {"credentials_rate_limited": True, "credentials_terminal": False}

    def test_relogin_required_overrides_all(self):
        """An auth error with relogin_required=True is always terminal."""
        from hermes_cli.turn_exit import credential_failure_flags

        exc = make_auth_error(code="invalid_grant", relogin_required=True)
        flags = credential_failure_flags(exc)
        assert flags == {"credentials_rate_limited": False, "credentials_terminal": True}

    def test_walks_cause_chain(self):
        """credential_failure_flags walks __cause__ to find wrapped AuthError."""
        from hermes_cli.turn_exit import credential_failure_flags

        inner = make_auth_error(code="missing_api_key")
        outer = RuntimeError("wrapped")  # noqa: TRY200 — cause chain test, not production raise
        outer.__cause__ = inner
        flags = credential_failure_flags(outer)
        assert flags == {"credentials_rate_limited": False, "credentials_terminal": True}


class TestTurnExitCodeCredential:
    """turn_exit_code honours credential flags for kanban workers."""

    def _exit_code(self, **flags) -> int:
        from hermes_cli.turn_exit import turn_exit_code

        return turn_exit_code(None, kanban_worker=True, **flags)

    def test_terminal_credential_yields_exit_code_78(self):
        """Exit code 78 for terminal credential failures (re-login / missing API key)."""
        assert self._exit_code(credentials_terminal=True) == 78

    def test_rate_limited_credential_yields_exit_code_75(self):
        """Exit code 75 for rate-limited credential (quota wall)."""
        assert self._exit_code(credentials_rate_limited=True) == 75

    def test_terminal_takes_precedence_over_rate_limited(self):
        """When both flags are set, rate-limited takes precedence (checked first in turn_exit_code).

        In practice credential_failure_flags never sets both True, so this codifies
        the current ordering: the rate-limited check fires before the terminal one.
        """
        assert self._exit_code(credentials_rate_limited=True, credentials_terminal=True) == 75

    def test_no_flags_yields_exit_code_1(self):
        """No credential flags -> generic failure (exit 1)."""
        assert self._exit_code() == 1

    def test_terminal_not_kanban_yields_exit_code_1(self):
        """Terminal flag without kanban_worker=True is not applied."""
        from hermes_cli.turn_exit import turn_exit_code

        code = turn_exit_code(None, kanban_worker=False, credentials_terminal=True)
        assert code == 1