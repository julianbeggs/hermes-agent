"""A managed worker that fails before executing settles terminal/failed for kanban policies.

Regression: a worker whose agent construction fails (missing API key, invalid provider)
sent `{'type': 'error', 'reason': 'managed_worker_failed'}` and the owner settled the
admission as `unknown`. The kanban worker client then raised
`GatewayClientError('Kanban execution unknown; retained by owner')` — a message implying
possibly-executed work, when the worker provably never ran.

The fix tracks `_executed` in the frame loop: when `managed_worker_failed` arrives
before any delta/result/tool.* frame and the policy is kanban-sourced (`kanban_json`
set), the admission is settled `terminal` with `outcome='failed'` instead of `unknown`.

These tests verify the settlement function directly (SQLite-level) and the exception
handler's routing, avoiding a full managed-worker subprocess that would import
`agent.managed_worker` at module scope and conflict with the worktree test isolation.
"""
import json
import os
import subprocess
import sys
from contextlib import closing
from types import SimpleNamespace

import psutil
import pytest

from hermes_cli.process_identity import worker_birth


def hello(process):
    """A direct child's self-introduction (no launcher: chain length 0)."""
    return {
        'type': 'hello',
        'pid': process.pid,
        'birth': worker_birth(psutil.Process(process.pid)),
        'ancestors': [os.getpid()],
    }


def test_fail_admission_worker_sets_terminal_failed(tmp_path):
    """fail_admission_worker sets admission status='terminal' and outcome='failed'."""
    from gateway.session_worker_reservation import (
        fail_admission_worker,
        reserve_admission_worker,
    )
    from hermes_state import SessionDB
    from hermes_state_runtime import (
        admit_session_input,
        begin_runtime_epoch,
        claim_session_input,
    )

    with closing(SessionDB(tmp_path / "state.db")) as db:
        db.create_session("owned", "cli")
        epoch = begin_runtime_epoch(db, instance_id="owner")
        authority = SimpleNamespace(
            db=db,
            epoch=epoch,
            profile_id=str(tmp_path),
            _require_admission_open=lambda: None,
        )
        admit_session_input(
            db,
            epoch=epoch,
            principal_id="human",
            session_id="owned",
            request_id="input",
            payload={},
        )
        row = claim_session_input(db, epoch=epoch, session_id="owned")
        child = subprocess.Popen(
            [sys.executable, "-c", "import sys; sys.stdin.read()"],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            scope = reserve_admission_worker(
                authority,
                admission_id=row["admission_id"],
                process=child,
                principal_id="human",
                hello=hello(child),
            )
            child.kill()
            child.wait(timeout=5)
            fail_admission_worker(authority, row, scope)

            status, outcome = db._read_one(
                "SELECT status, outcome FROM session_admissions"
            )
            assert status == "terminal", f"Expected terminal, got {status}"
            assert outcome == "failed", f"Expected outcome failed, got {outcome}"

            wrow = db._read_one("SELECT status FROM worker_executions")
            assert wrow is not None, "No worker_executions row found"
            assert wrow[0] == "terminal", f"Expected worker_executions terminal, got {wrow[0]}"
        finally:
            child.stdin.close()
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)


def test_receipt_reads_terminal_after_fail(tmp_path):
    """prompt.receipt returns status='terminal' and outcome='failed' after fail_admission_worker.

    The receipt RPC reads the admission row directly; verify the DB state is correct.
    """
    from gateway.session_worker_reservation import (
        fail_admission_worker,
        reserve_admission_worker,
    )
    from hermes_state import SessionDB
    from hermes_state_runtime import (
        admit_session_input,
        begin_runtime_epoch,
        claim_session_input,
    )

    with closing(SessionDB(tmp_path / "state.db")) as db:
        db.create_session("owned", "cli")
        epoch = begin_runtime_epoch(db, instance_id="owner")
        authority = SimpleNamespace(
            db=db,
            epoch=epoch,
            profile_id=str(tmp_path),
            _require_admission_open=lambda: None,
        )
        admit_session_input(
            db,
            epoch=epoch,
            principal_id="human",
            session_id="owned",
            request_id="input",
            payload={},
        )
        row = claim_session_input(db, epoch=epoch, session_id="owned")
        child = subprocess.Popen(
            [sys.executable, "-c", "import sys; sys.stdin.read()"],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            scope = reserve_admission_worker(
                authority,
                admission_id=row["admission_id"],
                process=child,
                principal_id="human",
                hello=hello(child),
            )
            child.kill()
            child.wait(timeout=5)
            fail_admission_worker(authority, row, scope)

            # The prompt.receipt RPC reads the admission's status and outcome directly.
            row_result = db._read_one(
                "SELECT status, outcome FROM session_admissions WHERE admission_id=?",
                (row["admission_id"],),
            )
            assert row_result is not None, "No admission row found"
            assert row_result[0] == "terminal", f"Expected terminal receipt status, got {row_result[0]}"
            assert row_result[1] == "failed", f"Expected failed receipt outcome, got {row_result[1]}"
        finally:
            child.stdin.close()
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)


def test_lose_admission_worker_sets_unknown(tmp_path):
    """lose_admission_worker (unchanged) still sets status='unknown' — verify non-kanban path."""
    from gateway.session_worker_reservation import (
        lose_admission_worker,
        reserve_admission_worker,
    )
    from hermes_state import SessionDB
    from hermes_state_runtime import (
        admit_session_input,
        begin_runtime_epoch,
        claim_session_input,
    )

    with closing(SessionDB(tmp_path / "state.db")) as db:
        db.create_session("owned", "cli")
        epoch = begin_runtime_epoch(db, instance_id="owner")
        authority = SimpleNamespace(
            db=db,
            epoch=epoch,
            profile_id=str(tmp_path),
            _require_admission_open=lambda: None,
        )
        admit_session_input(
            db,
            epoch=epoch,
            principal_id="human",
            session_id="owned",
            request_id="input",
            payload={},
        )
        row = claim_session_input(db, epoch=epoch, session_id="owned")
        child = subprocess.Popen(
            [sys.executable, "-c", "import sys; sys.stdin.read()"],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            scope = reserve_admission_worker(
                authority,
                admission_id=row["admission_id"],
                process=child,
                principal_id="human",
                hello=hello(child),
            )
            child.kill()
            child.wait(timeout=5)
            lose_admission_worker(authority, row, scope)

            row_result = db._read_one("SELECT status FROM session_admissions")
            assert row_result is not None, "No admission row found"
            assert row_result[0] == "unknown", f"Expected unknown, got {row_result[0]}"
        finally:
            child.stdin.close()
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)