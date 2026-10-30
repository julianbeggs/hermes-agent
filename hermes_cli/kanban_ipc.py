"""Kanban worker IPC — route worker DB writes through the dispatcher.

Architecture
------------
Workers spawned by the dispatcher write directly to the kanban SQLite DB for
heartbeats, completions, blocks, and log entries. When multiple workers run
concurrently, each opens its own connection and writes independently. Under
WAL journal mode, concurrent auto-checkpoints from separate connections can
race, producing a torn index B-tree — the root cause of the recurring
``idx_events_task`` corruption.

This module eliminates the multi-writer condition by routing all worker
write operations through a single-threaded Unix socket server running inside
the gateway dispatcher process. The dispatcher is the **sole writer** to the
kanban DB, and it serialises all writes on one connection.

Protocol
--------
Length-prefixed JSON messages over ``AF_UNIX SOCK_STREAM``.

Request (worker → dispatcher)::

    [4-byte big-endian payload length][JSON payload]

Response (dispatcher → worker)::

    [4-byte big-endian payload length][JSON response]

The JSON payload has the shape::

    {
        "action": "complete" | "block" | "heartbeat" | "log" | "comment",
        "task_id": "t_<hex>",
        "payload": { ... }
    }

The response is::

    {"ok": true, ...} | {"error": "..."}
"""

from __future__ import annotations

import json
import logging
import os
import socket
import struct
import threading
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Socket path resolution
# ---------------------------------------------------------------------------

SOCKET_DIR_TAG = "worker.sock"  # basename of the socket file


def _socket_path(board: Optional[str] = None) -> str:
    """Return the Unix socket path for a given board.

    Derives from the board's kanban DB path so multiple boards get
    independent sockets.
    """
    from hermes_cli.kanban_db import kanban_db_path

    db_path = kanban_db_path(board=board)
    return str(db_path.parent / SOCKET_DIR_TAG)


# ---------------------------------------------------------------------------
# Client side (called from worker processes)
# ---------------------------------------------------------------------------

def _is_worker_context() -> bool:
    """True when the current process is a dispatcher-spawned worker."""
    return bool(os.environ.get("HERMES_KANBAN_TASK"))


def send_worker_request(
    action: str,
    task_id: str,
    payload: dict[str, Any],
    *,
    board: Optional[str] = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Send a write request to the dispatcher IPC server and wait for a response.

    Parameters
    ----------
    action:
        One of ``"complete"``, ``"block"``, ``"heartbeat"``, ``"log"``,
        ``"comment"``.
    task_id:
        The kanban task id (``t_<hex>``).
    payload:
        Action-specific data. Keys and values depend on the action.
    board:
        Board slug. ``None`` uses the default board resolution chain.
    timeout:
        Seconds to wait for a response.

    Returns
    -------
    Parsed JSON response dict. Always contains ``"ok": True`` on success,
    or ``"error": "..."`` on failure.

    Raises
    ------
    ConnectionError
        If the socket is unreachable (dispatcher not running).
    TimeoutError
        If no response within *timeout* seconds.
    """
    sock_path = _socket_path(board=board)
    if not os.path.exists(sock_path):
        # Fallback: dispatcher IPC not available, connect directly.
        # This preserves backward compatibility for non-worker contexts
        # (e.g. CLI tools, tests) and for the transition period where
        # the dispatcher hasn't been updated yet.
        raise ConnectionError(
            f"IPC socket {sock_path} not found — dispatcher IPC server "
            f"is not running. Falling back to direct DB access."
        )

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect(sock_path)
        msg = json.dumps(
            {"action": action, "task_id": task_id, "payload": payload},
            separators=(",", ":"),
        ).encode("utf-8")
        # Length-prefixed send
        sock.sendall(struct.pack("!I", len(msg)) + msg)

        # Length-prefixed receive
        raw_len = _recv_exact(sock, 4)
        if not raw_len:
            raise ConnectionError("dispatcher closed connection before response")
        resp_len = struct.unpack("!I", raw_len)[0]
        raw = _recv_exact(sock, resp_len)
        if raw is None:
            raise ConnectionError("dispatcher closed connection mid-response")
        return json.loads(raw.decode("utf-8"))
    finally:
        try:
            sock.close()
        except Exception:
            pass


def _recv_exact(sock: socket.socket, n: int) -> Optional[bytes]:
    """Read exactly *n* bytes from *sock*.

    Returns ``None`` if the peer closes the connection before all bytes
    arrive.
    """
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


# ---------------------------------------------------------------------------
# Server side (runs inside the gateway dispatcher process)
# ---------------------------------------------------------------------------

class KanbanIpcServer:
    """A single-threaded Unix socket server for worker write requests.

    Each request is processed synchronously via the provided *dispatch_fn*,
    which receives ``(action, task_id, payload)`` and returns a JSON-serialisable
    dict. The dispatch function is called from the server thread, so it must
    be thread-safe with respect to its own resources (typically the kanban DB
    connection).

    The server runs as a daemon thread so it does not block gateway shutdown.
    """

    def __init__(
        self,
        board: Optional[str] = None,
    ):
        self._board = board
        self._sock_path = _socket_path(board=board)
        self._server: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._dispatch_fn: Optional[callable] = None

    def start(self, dispatch_fn: callable) -> threading.Thread:
        """Start the IPC server in a background daemon thread.

        *dispatch_fn* is called as ``dispatch_fn(action, task_id, payload)``
        and must return a dict that will be JSON-serialised as the response.
        """
        if self._thread is not None and self._thread.is_alive():
            logger.warning("kanban IPC server already running on %s", self._sock_path)
            return self._thread

        self._dispatch_fn = dispatch_fn

        # Remove any stale socket file from a previous run.
        try:
            os.unlink(self._sock_path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.warning("could not remove stale IPC socket %s: %s", self._sock_path, exc)

        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(self._sock_path)
        self._server.listen(5)
        os.chmod(self._sock_path, 0o600)

        self._thread = threading.Thread(
            target=self._serve_forever,
            name=f"kanban-ipc-{self._board or 'default'}",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "kanban IPC server started on %s (board=%s)",
            self._sock_path,
            self._board or "default",
        )
        return self._thread

    def stop(self) -> None:
        """Stop the IPC server and clean up the socket file."""
        if self._server is not None:
            try:
                self._server.close()
            except Exception:
                pass
            self._server = None
        try:
            os.unlink(self._sock_path)
        except FileNotFoundError:
            pass
        except OSError:
            pass

    def _serve_forever(self) -> None:
        """Accept connections and process requests until the server is closed."""
        if self._server is None:
            return
        while True:
            try:
                client, _addr = self._server.accept()
            except OSError:
                # Server socket closed during shutdown.
                break
            try:
                self._handle_client(client)
            except Exception:
                logger.exception("kanban IPC server: error handling client")
            finally:
                try:
                    client.close()
                except Exception:
                    pass

    def _handle_client(self, client: socket.socket) -> None:
        """Read one request, process it, and send the response."""
        raw_len = _recv_exact(client, 4)
        if not raw_len:
            return  # Client closed without sending anything
        msg_len = struct.unpack("!I", raw_len)[0]
        if msg_len > 1_000_000:  # 1 MB sanity cap
            self._send_error(client, "request payload too large")
            return
        raw = _recv_exact(client, msg_len)
        if raw is None:
            return
        try:
            msg = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            self._send_error(client, f"invalid JSON: {exc}")
            return

        action = msg.get("action")
        task_id = msg.get("task_id")
        payload = msg.get("payload", {})
        if not action or not task_id:
            self._send_error(client, "missing 'action' or 'task_id'")
            return

        fn = self._dispatch_fn
        if fn is None:
            self._send_error(client, "IPC server not ready (no dispatch function)")
            return

        try:
            result = fn(action, task_id, payload)
        except Exception as exc:
            logger.exception("kanban IPC: dispatch_fn raised for %s/%s", action, task_id)
            self._send_error(client, str(exc))
            return

        resp = json.dumps(result, separators=(",", ":")).encode("utf-8")
        try:
            client.sendall(struct.pack("!I", len(resp)) + resp)
        except Exception:
            logger.debug("kanban IPC: failed to send response to client")

    def _send_error(self, client: socket.socket, message: str) -> None:
        """Send an error response and close."""
        resp = json.dumps({"error": message}, separators=(",", ":")).encode("utf-8")
        try:
            client.sendall(struct.pack("!I", len(resp)) + resp)
        except Exception:
            pass