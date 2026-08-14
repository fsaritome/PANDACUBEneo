"""SQLite-backed processing ledger (§5.2).

Tracks per-file state so the backlog sweep is idempotent (skip files whose
content hash matches a prior successful run) and so QC data survives restarts.
A fresh connection is opened per call so this is safe to use from worker
processes as well as the main watcher process; WAL mode keeps concurrent
readers/writers from blocking each other on a single host.
"""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    input_path TEXT UNIQUE NOT NULL,
    output_path TEXT,
    content_hash TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',
    engines_used TEXT,
    confidence_summary TEXT,
    languages TEXT,
    layout_type TEXT,
    fallback_fired INTEGER NOT NULL DEFAULT 0,
    flagged INTEGER NOT NULL DEFAULT 0,
    flag_reason TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_files_status ON files(status);
CREATE INDEX IF NOT EXISTS idx_files_hash ON files(content_hash);
"""

# Terminal states for which a matching content hash means "nothing to do".
_STABLE_STATUSES = {"done", "flagged"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class FileRecord:
    input_path: str
    output_path: Optional[str]
    content_hash: str
    status: str
    engines_used: list[str]
    confidence_summary: dict
    languages: list[str]
    layout_type: Optional[str]
    fallback_fired: bool
    flagged: bool
    flag_reason: Optional[str]
    error: Optional[str]
    created_at: str
    updated_at: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "FileRecord":
        return cls(
            input_path=row["input_path"],
            output_path=row["output_path"],
            content_hash=row["content_hash"],
            status=row["status"],
            engines_used=json.loads(row["engines_used"] or "[]"),
            confidence_summary=json.loads(row["confidence_summary"] or "{}"),
            languages=json.loads(row["languages"] or "[]"),
            layout_type=row["layout_type"],
            fallback_fired=bool(row["fallback_fired"]),
            flagged=bool(row["flagged"]),
            flag_reason=row["flag_reason"],
            error=row["error"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


_WAL_SWITCH_RETRIES = 10
_WAL_SWITCH_BASE_DELAY = 0.05  # seconds


class Ledger:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        # Switching a freshly-created db into WAL mode requires an exclusive lock
        # that isn't retried by SQLite's busy_timeout, so multiple worker processes
        # opening the ledger at once can raise "database is locked" instantly.
        # Retry that specific pragma with backoff instead of one shot.
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        for attempt in range(_WAL_SWITCH_RETRIES):
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                break
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc) or attempt == _WAL_SWITCH_RETRIES - 1:
                    conn.close()
                    raise
                time.sleep(_WAL_SWITCH_BASE_DELAY * (attempt + 1))
        conn.execute("PRAGMA busy_timeout=30000")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def get(self, input_path: str) -> Optional[FileRecord]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM files WHERE input_path = ?", (input_path,)
            ).fetchone()
            return FileRecord.from_row(row) if row else None

    def needs_processing(self, input_path: str, content_hash: str) -> bool:
        """True if this file has no record, its hash changed, or a prior run failed."""
        rec = self.get(input_path)
        if rec is None:
            return True
        if rec.content_hash != content_hash:
            return True
        return rec.status not in _STABLE_STATUSES

    def enqueue(self, input_path: str, content_hash: str) -> None:
        """Insert or reset a row to 'queued', unless it's already done with this hash."""
        if not self.needs_processing(input_path, content_hash):
            return
        now = _now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO files (input_path, content_hash, status, created_at, updated_at)
                VALUES (?, ?, 'queued', ?, ?)
                ON CONFLICT(input_path) DO UPDATE SET
                    content_hash=excluded.content_hash,
                    status='queued',
                    error=NULL,
                    updated_at=excluded.updated_at
                """,
                (input_path, content_hash, now, now),
            )

    def mark_processing(self, input_path: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE files SET status='processing', updated_at=? WHERE input_path=?",
                (_now(), input_path),
            )

    def mark_done(
        self,
        input_path: str,
        output_path: str,
        *,
        engines_used: list[str],
        confidence_summary: dict[str, Any],
        languages: list[str],
        layout_type: Optional[str],
        fallback_fired: bool,
        flagged: bool = False,
        flag_reason: Optional[str] = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE files SET
                    output_path=?, status=?, engines_used=?, confidence_summary=?,
                    languages=?, layout_type=?, fallback_fired=?, flagged=?,
                    flag_reason=?, error=NULL, updated_at=?
                WHERE input_path=?
                """,
                (
                    output_path,
                    "flagged" if flagged else "done",
                    json.dumps(engines_used),
                    json.dumps(confidence_summary),
                    json.dumps(languages),
                    layout_type,
                    int(fallback_fired),
                    int(flagged),
                    flag_reason,
                    _now(),
                    input_path,
                ),
            )

    def mark_failed(self, input_path: str, error: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE files SET status='failed', error=?, updated_at=? WHERE input_path=?",
                (error, _now(), input_path),
            )

    def reconcile_interrupted(self) -> int:
        """Fail any row still marked 'processing' from a previous run.

        A SIGKILL (OOM killer, `kill -9`, host reboot) skips `process_file`'s
        except/finally, so its row stays 'processing' forever and the dashboard
        renders it as live work with an ever-growing elapsed time. Nothing else
        ever clears these. Callers must invoke this at startup, before any
        worker can legitimately set the status again.
        """
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE files SET status='failed', "
                "error='interrupted: process exited before completion', updated_at=? "
                "WHERE status='processing'",
                (_now(),),
            )
            return cur.rowcount or 0

    def iter_queued(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT input_path FROM files WHERE status='queued'"
            ).fetchall()
            return [r["input_path"] for r in rows]

    def flagged_report(self) -> list[FileRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM files WHERE flagged=1 ORDER BY updated_at DESC"
            ).fetchall()
            return [FileRecord.from_row(r) for r in rows]

    def status_counts(self) -> dict[str, int]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) AS n FROM files GROUP BY status"
            ).fetchall()
            return {r["status"]: r["n"] for r in rows}

    def all_records(self) -> list[FileRecord]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM files ORDER BY id").fetchall()
            return [FileRecord.from_row(r) for r in rows]

    def processing_records(self) -> list[FileRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM files WHERE status='processing' ORDER BY created_at"
            ).fetchall()
            return [FileRecord.from_row(r) for r in rows]

    def failed_records(self) -> list[FileRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM files WHERE status='failed' ORDER BY updated_at DESC"
            ).fetchall()
            return [FileRecord.from_row(r) for r in rows]
