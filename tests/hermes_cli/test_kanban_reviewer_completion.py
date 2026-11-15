"""Reviewer completions must persist a machine-readable verdict or stay retryable."""
from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def conn(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    with kbc.connect() as connection:
        yield connection


def _claimed_task(conn, assignee="code-reviewer") -> str:
    tid = kb.create_task(conn, title="review verdict contract", assignee=assignee)
    assert kb.claim_task(conn, tid, claimer=kb._claimer_id()) is not None
    return tid


def _changes_requested_metadata():
    return {
        "verdict": "changes_requested",
        "blocking_count": 2,
        "findings": [
            {"severity": "critical", "file": "one.py", "line": 1,
             "issue": "first", "suggested_fix": "fix one"},
            {"severity": "critical", "file": "two.py", "line": 2,
             "issue": "second", "suggested_fix": "fix two"},
        ],
    }


def test_empty_reviewer_completion_is_rejected_then_retryable(conn):
    tid = _claimed_task(conn)
    run_id = kb._current_run_id(conn, tid)

    with pytest.raises(kb.ReviewerCompletionError, match="verdict"):
        kb.complete_task(conn, tid, summary="changes requested", metadata={})

    task = kb.get_task(conn, tid)
    assert task.status == "running"
    assert task.current_run_id == run_id
    assert not any(event.kind == "completed" for event in kb.list_events(conn, tid))

    metadata = _changes_requested_metadata()
    assert kb.complete_task(conn, tid, summary="changes requested", metadata=metadata) is True
    persisted = kb.latest_run(conn, tid).metadata
    assert persisted["verdict"] == "changes_requested"
    assert persisted["blocking_count"] == 2
    assert persisted["findings"] == metadata["findings"]


@pytest.mark.parametrize("metadata", [
    None,
    {},
    {"verdict": "looks_good", "blocking_count": 0, "findings": []},
    {"verdict": "approved", "findings": []},
    {"verdict": "approved", "blocking_count": True, "findings": []},
    {"verdict": "approved", "blocking_count": -1, "findings": []},
    {"verdict": "approved", "blocking_count": 0},
    {"verdict": "approved", "blocking_count": 0, "findings": {}},
])
def test_invalid_reviewer_contract_never_completes(conn, metadata):
    tid = _claimed_task(conn)
    with pytest.raises(kb.ReviewerCompletionError):
        kb.complete_task(conn, tid, summary="reviewed", metadata=metadata)
    assert kb.get_task(conn, tid).status == "running"


def test_valid_approval_contract_still_completes(conn):
    tid = _claimed_task(conn)
    metadata = {"verdict": "approved", "blocking_count": 0, "findings": []}
    assert kb.complete_task(conn, tid, summary="approved", metadata=metadata) is True
    assert kb.latest_run(conn, tid).metadata == metadata


def test_coder_completion_does_not_inherit_reviewer_contract(conn):
    tid = _claimed_task(conn, assignee="coder")
    assert kb.complete_task(conn, tid, summary="implemented", metadata={"files": 2}) is True
    assert kb.latest_run(conn, tid).metadata == {"files": 2}
