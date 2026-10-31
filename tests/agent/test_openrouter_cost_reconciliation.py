"""OpenRouter true-cost reconciliation contracts (agent/openrouter_cost.py).

``openrouter/auto`` reports a sentinel cost, so finalize kicks a reconciliation against
OpenRouter's activity/credits endpoints. Contracts verified here: matched activity rows
update the session with ``cost_status='reconciled'`` / ``cost_source='openrouter_api'``;
any API failure leaves the prior estimated/unknown status untouched; the snapshot gate
only fires for OpenRouter-billed turns with placeholder accounting.
"""

import threading

import pytest

from agent import openrouter_cost
from agent.openrouter_cost import (
    RECONCILED_COST_SOURCE,
    RECONCILED_COST_STATUS,
    capture_turn_reconciliation_snapshot,
    fetch_credit_balance,
    match_session_rows,
    maybe_schedule_reconciliation,
    reconcile_session_cost,
)
from tests.waiting import wait_until
from datetime import UTC

_NOW = 1_800_000_000.0


class _RecordingStore:
    def __init__(self):
        self.queued = []

    def queue_token_counts(self, session_id, **usage):
        self.queued.append((session_id, usage))


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        import json
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _activity_row(*, model="vendor/routed-model", cost=0.0123, prompt=100, completion=50,
                  time_offset=0.0):
    return {
        "id": "gen-1",
        "model": model,
        "cost": cost,
        "tokens_prompt": prompt,
        "tokens_completion": completion,
        "time": _NOW + time_offset,
    }


def test_matched_rows_reconcile_cost_status():
    store = _RecordingStore()
    rows = [_activity_row(cost=0.0123, prompt=100, completion=50)]
    total = reconcile_session_cost(
        store.queue_token_counts, "sess-1", api_key="or-key", model="vendor/routed-model",
        start_epoch=_NOW - 10, end_epoch=_NOW + 10,
        input_tokens=100, output_tokens=50,
        fetch_rows=lambda api_key: rows,
    )
    assert total == pytest.approx(0.0123)
    assert store.queued == [(
        "sess-1",
        {"actual_cost_usd": pytest.approx(0.0123), "cost_status": RECONCILED_COST_STATUS,
         "cost_source": RECONCILED_COST_SOURCE},
    )]


def test_api_failure_leaves_prior_status_untouched():
    store = _RecordingStore()

    def _boom(api_key):
        raise OSError("network down")

    total = reconcile_session_cost(
        store.queue_token_counts, "sess-1", api_key="or-key", model="vendor/routed-model",
        start_epoch=_NOW - 10, end_epoch=_NOW + 10,
        fetch_rows=_boom,
    )
    assert total is None
    assert store.queued == []  # estimated/unknown row untouched


def test_match_filters_window_and_model():
    rows = [
        _activity_row(model="other/model"),
        _activity_row(time_offset=-10000.0),  # outside the window
        _activity_row(time_offset=30.0, cost=0.5),
    ]
    matched = match_session_rows(
        rows, model="vendor/routed-model", start_epoch=_NOW - 10, end_epoch=_NOW + 10)
    assert [row["cost"] for row in matched] == [0.5]


def test_match_prefers_exact_token_counts():
    rows = [
        _activity_row(prompt=100, completion=50, cost=0.0123),
        _activity_row(prompt=999, completion=999, cost=4.0),
    ]
    matched = match_session_rows(
        rows, model="vendor/routed-model", start_epoch=_NOW - 10, end_epoch=_NOW + 10,
        input_tokens=100, output_tokens=50)
    assert len(matched) == 1 and matched[0]["cost"] == 0.0123


def test_iso_timestamp_row_matches():
    from datetime import datetime, timezone
    row = dict(_activity_row())
    row["time"] = datetime.fromtimestamp(_NOW + 5, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    matched = match_session_rows(
        [row], model="vendor/routed-model", start_epoch=_NOW - 10, end_epoch=_NOW + 10)
    assert len(matched) == 1


def test_capture_snapshot_gates():
    class _Agent:
        provider = "openrouter"
        base_url = "https://openrouter.ai/api/v1"
        model = "openrouter/auto"
        api_key = "or-key"
        session_cost_status = "estimated"

    snapshot = capture_turn_reconciliation_snapshot(
        _Agent(), start_epoch=_NOW - 5, end_epoch=_NOW, input_tokens=10, output_tokens=2)
    assert snapshot is not None and snapshot["api_key"] == "or-key"

    _Agent.session_cost_status = "provider"
    assert capture_turn_reconciliation_snapshot(
        _Agent(), start_epoch=_NOW, end_epoch=_NOW) is None

    _Agent.session_cost_status = "estimated"
    _Agent.provider = "openai"
    _Agent.base_url = "https://api.openai.com/v1"
    assert capture_turn_reconciliation_snapshot(
        _Agent(), start_epoch=_NOW, end_epoch=_NOW) is None


def test_schedule_skips_without_snapshot():
    store = _RecordingStore()
    before = threading.active_count()
    maybe_schedule_reconciliation(store, "sess-1", None)
    assert threading.active_count() == before  # nothing scheduled


def test_schedule_runs_reconciliation_on_daemon_thread():
    store = _RecordingStore()
    done = threading.Event()
    original = openrouter_cost.reconcile_session_cost

    def _rows(api_key):
        done.set()
        return [_activity_row(cost=0.25)]

    # Thread path uses the real fetch; stub the module-level fetcher it calls.
    def _reconcile(queue, session_id, **kwargs):
        return original(queue, session_id, fetch_rows=_rows, **kwargs)

    real_schedule = openrouter_cost.maybe_schedule_reconciliation

    from unittest import mock

    snapshot = {"api_key": "k", "model": "vendor/routed-model", "start_epoch": _NOW - 10,
                "end_epoch": _NOW + 10, "input_tokens": 100, "output_tokens": 50,
                "cost_status": "estimated"}
    with mock.patch.object(openrouter_cost, "reconcile_session_cost", _reconcile), \
            mock.patch.object(openrouter_cost, "fetch_credit_balance", lambda key: 5.0):
        real_schedule(store, "sess-1", snapshot)
        assert wait_until(done.wait, timeout=5.0)
        assert wait_until(lambda: bool(store.queued), timeout=5.0)
        usage = store.queued[0][1]
        assert usage["cost_status"] == RECONCILED_COST_STATUS
        assert usage["cost_source"] == RECONCILED_COST_SOURCE
        assert usage["actual_cost_usd"] == pytest.approx(0.25)


def test_fetch_credit_balance_parses_and_fails_soft():
    payload = {"data": {"total_credits": 10.0, "total_usage": 2.5}}
    assert fetch_credit_balance("k", fetch=lambda req, timeout: _FakeResponse(payload)) == 7.5

    def _fail(req, timeout):
        raise OSError("down")

    assert fetch_credit_balance("k", fetch=_fail) is None
    assert fetch_credit_balance("", fetch=_fail) is None
