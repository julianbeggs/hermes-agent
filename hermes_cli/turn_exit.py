"""Turn-result exit policy shared by CLI views and authority-owned workers.

Worker identity is supplied by the caller, not inferred from a daemon's environment.
"""

# Provider walls do not spend a Kanban task's retry budget.
TRANSIENT_PROVIDER_REASONS = frozenset({
    "rate_limit", "upstream_rate_limit", "billing", "overloaded", "server_error", "timeout",
})
# These failures need an operator change; the dispatcher parks the task immediately.
TERMINAL_PROVIDER_REASONS = frozenset({
    "auth", "auth_permanent", "model_not_found", "ssl_cert_verification", "upstream_blocked",
})
# AuthError codes indicating the provider's credentials are missing or invalid — the operator
# must configure/authenticate before retrying. Distinct from rate-limit/quota codes which are
# transient and from entitlement/refresh codes that may resolve without operator intervention.
CREDENTIAL_MISSING_CODES = frozenset({
    "missing_api_key",
    "no_provider_configured",
    "invalid_provider",
    "nous_auth_missing",
    "qwen_auth_missing",
    "codex_auth_missing",
    "codex_auth_invalid_shape",
    "corrupt_config",
})


def turn_exit_code(
    result, *, kanban_worker: bool, credentials_rate_limited: bool = False,
    credentials_terminal: bool = False,
    transient_reasons=TRANSIENT_PROVIDER_REASONS, terminal_reasons=TERMINAL_PROVIDER_REASONS,
) -> int:
    """Preserve completion, interruption and retryable/terminal provider outcomes."""
    if not isinstance(result, dict):
        if credentials_rate_limited and kanban_worker:
            from hermes_cli.kanban_db import KANBAN_RATE_LIMIT_EXIT_CODE
            return KANBAN_RATE_LIMIT_EXIT_CODE
        if credentials_terminal and kanban_worker:
            from hermes_cli.kanban_db import KANBAN_TERMINAL_PROVIDER_EXIT_CODE
            return KANBAN_TERMINAL_PROVIDER_EXIT_CODE
        return 1
    if result.get("interrupted"):
        return 130
    if not (result.get("failed") or result.get("partial") or result.get("completed") is False):
        return 0
    if kanban_worker:
        reason = result.get("failure_reason")
        if reason in transient_reasons:
            from hermes_cli.kanban_db import KANBAN_RATE_LIMIT_EXIT_CODE
            return KANBAN_RATE_LIMIT_EXIT_CODE
        if reason in terminal_reasons:
            from hermes_cli.kanban_db import KANBAN_TERMINAL_PROVIDER_EXIT_CODE
            return KANBAN_TERMINAL_PROVIDER_EXIT_CODE
    return 1


def credential_failure_flags(exc) -> dict:
    """``turn_exit_code`` credential kwargs for an exception raised before any turn result exists.

    A quota/rate-limit ``AuthError`` is transient (exit code 75).
    An explicit re-authentication requirement *or* a
    ``CREDENTIAL_MISSING_CODES`` code (missing/invalid provider configuration) is terminal
    (exit code 78). The cause chain is walked because agent construction wraps the resolver's
    error."""
    from hermes_cli.auth import AuthError, is_rate_limited_auth_error
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, AuthError):
            rate_limited = is_rate_limited_auth_error(exc)
            terminal = bool(exc.relogin_required) or (
                not rate_limited and exc.code in CREDENTIAL_MISSING_CODES)
            return {"credentials_rate_limited": rate_limited,
                    "credentials_terminal": terminal}
        exc = exc.__cause__ or exc.__context__
    return {}
