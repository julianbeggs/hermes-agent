"""A managed worker that fails before executing settles terminal/failed for kanban policies.

Regression: a worker whose agent construction fails (missing API key, invalid provider)
sent `{'type': 'error', 'reason': 'managed_worker_failed'}` and the owner settled the
admission as `unknown`. The kanban worker client then raised
`GatewayClientError('Kanban execution unknown; retained by owner')` — a message implying
possibly-executed work, when the worker provably never ran.

The fix tracks `_executed` in the frame loop: when `managed_worker_failed` arrives
before any delta/result/tool.* frame and the policy is kanban-sourced (`kanban_json`
set), the admission is settled `terminal` with `outcome='failed'` instead of `unknown`.
"""
import asyncio
import dataclasses
import json
import os
import sqlite3
import subprocess
import sys
from types import SimpleNamespace

import psutil
import pytest

from gateway.session_contract import SessionActor, SessionRef
from hermes_cli.process_identity import worker_birth

# A minimal child that:
# 1. sends 'hello'
# 2. reads bootstrap (then ignores it)
# 3. sends 'error'/'managed_worker_failed' — simulating agent construction failure
CHILD_PRE_EXEC_FAILURE = """\
import json, os, sys, psutil
from hermes_cli.process_identity import worker_birth
me = psutil.Process()
out = os.fdopen(os.dup(sys.stdout.fileno()), 'wb', buffering=0)
out.write((json.dumps({'type': 'hello', 'pid': me.pid, 'birth': worker_birth(me),
                       'ancestors': [p.pid for p in me.parents()[:3]]}) + '\\n').encode())
out.flush()
# Read bootstrap (ignore it)
_ = sys.stdin.buffer.readline()
# Simulate agent construction failure
out.write(b'{"type": "error", "reason": "managed_worker_failed"}\\n')
out.flush()
"""


def _authority(monkeypatch, tmp_path, kanban_json: str | None):
    """Build a SessionAuthority and its managed turn fixture.

    Returns (authority, ref, policy, claim, published, db) — same shape as
    test_managed_worker_unknown_status's helper but with a pre-exec-failure child
    and an optional ``kanban_json`` on the policy.
    """
    from gateway import session_managed_worker as managed
    from gateway.session_authority import LiveSession, SessionAuthority
    from gateway.session_contract import Submission
    from gateway.session_policy import build_policy
    from hermes_state import SessionDB
    from hermes_state_runtime import begin_runtime_epoch, claim_session_input

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("s", source="test")
    runner = SimpleNamespace(
        _draining=False,
        _cached_agent_for=lambda route: None,
        _adapter_for_source=lambda source: None,
    )
    authority = SessionAuthority(
        runner,
        profile_id="owned",
        instance_id="i",
        db=db,
        epoch=begin_runtime_epoch(db, instance_id="i"),
    )
    live = authority.sessions["s"] = LiveSession(
        SimpleNamespace(platform=None, user_id="human", chat_id="c"), "route"
    )
    published = []
    original = live.event_stream.publish
    monkeypatch.setattr(
        live.event_stream,
        "publish",
        lambda sid, payload, **kw: (
            published.append((kw.get("event_type", "message.complete"), payload)),
            original(sid, payload, **kw),
        )[1],
    )
    actor = SessionActor("human", "owned", frozenset({"session:submit", "session:control"}), "cli")
    ref = SessionRef("owned", "s")
    monkeypatch.setattr(authority, "_schedule", lambda ref: None)

    class Child(subprocess.Popen):
        def __init__(self, args, **kw):
            super().__init__([sys.executable, "-c", CHILD_PRE_EXEC_FAILURE], **kw)

    monkeypatch.setattr(managed.subprocess, "Popen", Child)

    policy = build_policy({"cwd": str(tmp_path), "model": "m"}, {}, profile_terminal=False)
    if kanban_json is not None:
        policy = dataclasses.replace(policy, kanban_json=kanban_json)

    async def claim():
        await authority.submit(actor, Submission("turn", ref, {"text": "go"}, "queue"))
        return claim_session_input(db, epoch=authority.epoch, session_id="s")

    return authority, ref, policy, claim, published, db


@pytest.mark.platforms("linux")
def test_pre_exec_failure_with_kanban_settles_terminal(tmp_path, monkeypatch):
    """A kanban worker that fails before execution settles terminal/failed, not unknown."""
    from gateway import session_managed_worker as managed

    kanban_ctx = json.dumps({"task_id": "t_pre_exec", "db": str(tmp_path / "kanban.db"), "run_id": 1})
    authority, ref, policy, claim, published, db = _authority(monkeypatch, tmp_path, kanban_json=kanban_ctx)

    async def scenario():
        row = await claim()
        authority.executing[ref.session_id] = row["generation"]
        with pytest.raises(managed.ManagedExecutionUnknown):
            async with asyncio.timeout(30):
                await managed.execute_managed(authority, ref, row, policy)
        return row

    with db:
        row = asyncio.run(scenario())

    # Admission must be 'terminal', not 'unknown'
    with sqlite3.connect(tmp_path / "state.db") as ledger:
        status, outcome = ledger.execute(
            "SELECT status, outcome FROM session_admissions WHERE admission_id=?",
            (row["admission_id"],),
        ).fetchone()
    assert status == "terminal", f"Expected terminal, got {status}"
    assert outcome == "failed", f"Expected outcome failed, got {outcome}"

    # Worker execution must be 'terminal'
    with sqlite3.connect(tmp_path / "state.db") as ledger:
        wstatus = ledger.execute(
            "SELECT status FROM worker_executions"
        ).fetchone()[0]
    assert wstatus == "terminal", f"Expected worker_executions terminal, got {wstatus}"

    # Published event must carry outcome 'failed'
    complete = [
        p for kind, p in published
        if kind == "message.complete" and "outcome" in p
    ]
    assert complete, "No message.complete events with outcome published"
    failed = [p for p in complete if p["outcome"] == "failed"]
    assert failed, f"No 'failed' outcome among published: {[p.get('outcome') for p in complete]}"


@pytest.mark.platforms("linux")
def test_pre_exec_failure_without_kanban_settles_unknown(tmp_path, monkeypatch):
    """A NON-kanban worker that fails before execution keeps the existing 'unknown' settlement."""
    from gateway import session_managed_worker as managed

    authority, ref, policy, claim, published, db = _authority(monkeypatch, tmp_path, kanban_json=None)

    async def scenario():
        row = await claim()
        authority.executing[ref.session_id] = row["generation"]
        with pytest.raises(managed.ManagedExecutionUnknown):
            async with asyncio.timeout(30):
                await managed.execute_managed(authority, ref, row, policy)
        return row

    with db:
        row = asyncio.run(scenario())

    # Admission must be 'unknown' (unchanged behaviour for non-kanban)
    with sqlite3.connect(tmp_path / "state.db") as ledger:
        status = ledger.execute(
            "SELECT status FROM session_admissions WHERE admission_id=?",
            (row["admission_id"],),
        ).fetchone()[0]
    assert status == "unknown", f"Expected unknown for non-kanban, got {status}"


@pytest.mark.platforms("linux")
def test_pre_exec_failure_receipt_reads_terminal(tmp_path, monkeypatch):
    """prompt.receipt RPC returns status='terminal' for a pre-exec kanban failure."""
    from gateway import session_managed_worker as managed

    kanban_ctx = json.dumps({"task_id": "t_receipt", "db": str(tmp_path / "kanban.db"), "run_id": 1})
    authority, ref, policy, claim, published, db = _authority(monkeypatch, tmp_path, kanban_json=kanban_ctx)

    async def scenario():
        row = await claim()
        authority.executing[ref.session_id] = row["generation"]
        with pytest.raises(managed.ManagedExecutionUnknown):
            async with asyncio.timeout(30):
                await managed.execute_managed(authority, ref, row, policy)
        # Simulate the receipt read
        receipt = await authority.receipt(
            SessionActor("human", "owned", frozenset({"session:submit"}), "cli"),
            ref,
            row["admission_id"],
        )
        return receipt

    with db:
        receipt = asyncio.run(scenario())

    assert receipt.status == "terminal", f"Expected receipt status terminal, got {receipt.status}"
    assert receipt.outcome == "failed", f"Expected receipt outcome failed, got {receipt.outcome}"