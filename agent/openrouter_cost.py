"""OpenRouter true-cost reconciliation for finalized sessions.

``openrouter/auto`` is a router: the usage object it returns prices a sentinel, not the
upstream model that actually served the request, so the session's estimated cost is only
a placeholder. After a session (or managed-worker turn) finalizes, this module queries
OpenRouter's usage/credits endpoints with the session's own key and reconciles the real
per-call costs into state.db (``cost_status='reconciled'``,
``cost_source='openrouter_api'``).

Everything here is fail-soft by contract: any network, auth or parse problem logs at
debug and leaves the existing ``estimated``/``unknown`` cost accounting untouched. It
must never block or slow worker exit, so the reconciliation runs on a daemon thread.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from hermes_constants import OPENROUTER_BASE_URL

logger = logging.getLogger("agent.openrouter_cost")

_CREDITS_URL = f"{OPENROUTER_BASE_URL}/credits"
_ACTIVITY_URL = f"{OPENROUTER_BASE_URL}/activity"

RECONCILED_COST_STATUS = "reconciled"
RECONCILED_COST_SOURCE = "openrouter_api"
# Only placeholder accounting is worth reconciling; a session already priced by the
# provider (or included) must never be overwritten.
_RECONCILABLE_COST_STATUSES = frozenset({"estimated", "unknown"})

# OpenRouter stamps activity rows at request completion; clock skew and ingestion lag
# mean the session's own wall clock brackets the rows only with slack on both ends.
_WINDOW_SLACK_SECONDS = 120.0
_HTTP_TIMEOUT_SECONDS = 15.0


def _parse_epoch(value: Any) -> Optional[float]:
    """Accept an epoch (s or ms) or an ISO-8601 timestamp and return epoch seconds."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        epoch = float(value)
        return epoch / 1000.0 if epoch > 1e11 else epoch  # ms -> s
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            epoch = float(text)
            return epoch / 1000.0 if epoch > 1e11 else epoch
        except ValueError:
            pass
        try:
            return datetime.fromisoformat(text).timestamp()
        except ValueError:
            return None
    return None


def _row_time(row: dict[str, Any]) -> Optional[float]:
    return _parse_epoch(row.get("time") if row.get("time") is not None else row.get("timestamp"))


def _row_cost(row: dict[str, Any]) -> Optional[float]:
    """Activity rows report spend as a positive USD cost."""
    cost = row.get("cost")
    if not isinstance(cost, (int, float)) or isinstance(cost, bool):
        return None
    return abs(float(cost))


def _row_tokens(row: dict[str, Any]) -> tuple[int, int]:
    prompt = row.get("tokens_prompt")
    completion = row.get("tokens_completion")
    if not isinstance(prompt, int) or not isinstance(completion, int):
        usage = row.get("usage")
        if isinstance(usage, dict):
            prompt = prompt if isinstance(prompt, int) else usage.get("prompt_tokens")
            completion = completion if isinstance(completion, int) else usage.get("completion_tokens")
    return (prompt or 0) if isinstance(prompt, int) else 0, (completion or 0) if isinstance(completion, int) else 0


def _model_matches(row_model: Any, session_model: str) -> bool:
    """OpenRouter reports the routed model (``vendor/name``); the session may request it
    directly or via a router alias, so match on the normalized slug."""
    if not isinstance(row_model, str) or not row_model or not session_model:
        return False
    row_slug = row_model.strip().lower()
    session_slug = session_model.strip().lower()
    return row_slug == session_slug or row_slug.endswith("/" + session_slug) or session_slug.endswith("/" + row_slug)


def match_session_rows(
    rows: list[dict[str, Any]], *, model: str, start_epoch: float, end_epoch: float,
    input_tokens: Optional[int] = None, output_tokens: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Correlate OpenRouter activity rows to this session: timestamp window, then model;
    when the caller knows the turn's token counts, only rows that add up to them qualify
    (the strongest signal when a key serves several concurrent sessions)."""
    window_start = start_epoch - _WINDOW_SLACK_SECONDS
    window_end = end_epoch + _WINDOW_SLACK_SECONDS
    candidates = [
        row for row in rows
        if (t := _row_time(row)) is not None and window_start <= t <= window_end
        and _model_matches(row.get("model"), model)
    ]
    if input_tokens is None and output_tokens is None:
        return candidates
    expected = (input_tokens or 0, output_tokens or 0)
    exact = [row for row in candidates if _row_tokens(row) == expected]
    if exact:
        return exact
    # Token counts drifted (retries, cache accounting): keep the window+model match so the
    # reconciliation still lands, cost accuracy is bounded by the key's own activity log.
    return candidates


def fetch_credit_balance(api_key: str, *, timeout: float = _HTTP_TIMEOUT_SECONDS, fetch: Optional[Callable[..., Any]] = None) -> Optional[float]:
    """Remaining OpenRouter credit balance (``total_credits - total_usage``), or None."""
    from urllib.error import HTTPError, URLError
    from urllib.request import Request, urlopen

    if not api_key:
        return None
    request = Request(_CREDITS_URL, headers={"Authorization": f"Bearer {api_key}"})
    fetch = fetch or urlopen
    try:
        with fetch(request, timeout=timeout) as response:
            import json

            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, ValueError, OSError) as exc:
        logger.debug("OpenRouter credits fetch failed: %s", exc)
        return None
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, dict):
        return None
    total_credits = data.get("total_credits")
    total_usage = data.get("total_usage")
    if not isinstance(total_credits, (int, float)) or not isinstance(total_usage, (int, float)):
        return None
    return float(total_credits) - float(total_usage)


def fetch_usage_rows(api_key: str, *, timeout: float = _HTTP_TIMEOUT_SECONDS, fetch: Optional[Callable[..., Any]] = None) -> list[dict[str, Any]]:
    """Per-call usage rows from OpenRouter's activity endpoint (fail-soft: [] on error)."""
    from urllib.error import HTTPError, URLError
    from urllib.request import Request, urlopen

    if not api_key:
        return []
    request = Request(_ACTIVITY_URL, headers={"Authorization": f"Bearer {api_key}"})
    fetch = fetch or urlopen
    try:
        with fetch(request, timeout=timeout) as response:
            import json

            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, ValueError, OSError) as exc:
        logger.debug("OpenRouter activity fetch failed: %s", exc)
        return []
    data = payload.get("data") if isinstance(payload, dict) else None
    return data if isinstance(data, list) else []


def reconcile_session_cost(
    queue, session_id: str, *, api_key: str, model: str, start_epoch: float, end_epoch: float,
    input_tokens: Optional[int] = None, output_tokens: Optional[int] = None,
    fetch_rows: Optional[Callable[..., list[dict[str, Any]]]] = None,
) -> Optional[float]:
    """Sum the OpenRouter activity rows matched to this session/turn and queue the result
    onto ``session_id`` (``cost_status='reconciled'``, ``cost_source='openrouter_api'``).
    Returns the reconciled USD total, or None when nothing could be reconciled. Never
    raises: a placeholder status is better than a lost turn."""
    fetch_rows = fetch_rows or fetch_usage_rows
    try:
        rows = fetch_rows(api_key)
    except Exception as exc:  # health: allow BLE001 -- fail-soft boundary: an API error must keep the estimate as-is
        logger.debug("OpenRouter reconciliation fetch failed (session=%s): %s", session_id, exc, exc_info=True)
        return None
    matched = match_session_rows(
        rows, model=model, start_epoch=start_epoch, end_epoch=end_epoch,
        input_tokens=input_tokens, output_tokens=output_tokens,
    )
    total = 0.0
    for row in matched:
        cost = _row_cost(row)
        if cost is not None:
            total += cost
    if not matched or total <= 0:
        logger.debug("OpenRouter reconciliation matched nothing (session=%s, model=%s)", session_id, model)
        return None
    try:
        queue(session_id, actual_cost_usd=total, cost_status=RECONCILED_COST_STATUS,
              cost_source=RECONCILED_COST_SOURCE)
    except Exception as exc:  # health: allow BLE001 -- fail-soft boundary: a lost write must not fail the turn
        logger.debug("OpenRouter reconciliation write failed (session=%s): %s", session_id, exc, exc_info=True)
        return None
    return total


def capture_turn_reconciliation_snapshot(
    agent: Any, *, start_epoch: float, end_epoch: float,
    input_tokens: Optional[int] = None, output_tokens: Optional[int] = None,
) -> Optional[dict[str, Any]]:
    """Snapshot everything reconciliation needs from a live agent BEFORE it is retired.
    None unless this turn actually billed through OpenRouter with placeholder accounting."""
    base_url = str(getattr(agent, "base_url", "") or "")
    provider = str(getattr(agent, "provider", "") or "")
    from utils import base_url_host_matches

    if not (base_url_host_matches(base_url, "openrouter.ai") or provider == "openrouter"):
        return None
    status = getattr(agent, "session_cost_status", None)
    if status not in _RECONCILABLE_COST_STATUSES:
        return None
    api_key = str(getattr(agent, "api_key", "") or "").strip()
    if not api_key:
        return None
    return {
        "model": getattr(agent, "model", "") or "",
        "api_key": api_key,
        "start_epoch": start_epoch,
        "end_epoch": end_epoch,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_status": status,
    }


def maybe_schedule_reconciliation(store: Any, session_id: str, snapshot: Optional[dict[str, Any]]) -> None:
    """Fire-and-forget reconciliation after finalize: a daemon thread, fully fail-soft —
    an API error or a closed store leaves the prior cost status untouched and never
    blocks worker exit."""
    if not snapshot or not session_id:
        return
    if snapshot.get("cost_status") not in _RECONCILABLE_COST_STATUSES:
        return

    from agent.memory_provider import spawn_context_thread

    def _run() -> None:
        try:
            total = reconcile_session_cost(
                store.queue_token_counts, session_id,
                api_key=snapshot["api_key"], model=snapshot["model"],
                start_epoch=snapshot["start_epoch"], end_epoch=snapshot["end_epoch"],
                input_tokens=snapshot.get("input_tokens"), output_tokens=snapshot.get("output_tokens"),
            )
            if total is not None:
                balance = fetch_credit_balance(snapshot["api_key"])
                logger.info(
                    "OpenRouter cost reconciled (session=%s): $%.6f (%s)%s", session_id, total,
                    RECONCILED_COST_SOURCE,
                    f"; credits remaining ${balance:.2f}" if balance is not None else "",
                )
        except Exception as exc:  # health: allow BLE001 -- never block worker exit on accounting
            logger.debug("OpenRouter reconciliation failed (session=%s): %s", session_id, exc, exc_info=True)

    spawn_context_thread(_run, name="openrouter-cost-reconcile").start()
