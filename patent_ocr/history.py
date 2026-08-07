"""Persistent run history (§ dashboard).

Every `sweep` archives a summary row into a *separate* SQLite file
(`state/history.sqlite3` by default) that is never touched by the
input/output/qc/failed/ledger reset routine used between benchmark runs, so
long-term stats survive across runs.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Optional

from patent_ocr.config import Config
from patent_ocr.ledger import Ledger
from patent_ocr.qc import aggregate_qc_dir

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    finished_at TEXT NOT NULL,
    duration_seconds REAL NOT NULL,
    total_files INTEGER NOT NULL,
    done_count INTEGER NOT NULL,
    failed_count INTEGER NOT NULL,
    flagged_count INTEGER NOT NULL,
    passthrough_count INTEGER NOT NULL,
    skip_text_count INTEGER NOT NULL,
    ocr_count INTEGER NOT NULL,
    mean_confidence REAL,
    engine_win_counts TEXT NOT NULL,
    failed_files TEXT NOT NULL
);
"""


def history_db_path(config: Config) -> Path:
    return config.ledger_path.parent / "history.sqlite3"


@contextmanager
def _connect(db_path: Path) -> Iterator[sqlite3.Connection]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=30000")
    conn.executescript(_SCHEMA)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _classify_qc(qc_path: Path) -> str:
    """Categorize one file's qc sidecar: 'passthrough' | 'skip_text' | 'ocr'."""
    summary = json.loads(qc_path.read_text(encoding="utf-8"))
    if summary.get("passthrough"):
        return "passthrough"
    if not summary.get("pages"):
        return "skip_text"
    return "ocr"


def record_run(config: Config) -> dict[str, Any]:
    """Summarize the just-finished sweep (current ledger + qc dir contents)
    and persist it as one row in the history db. Safe to call even if some
    files are still queued/processing (e.g. after a Ctrl+C)."""
    ledger = Ledger(config.ledger_path)
    rows = ledger.all_records()

    if not rows:
        return {}

    started_at = min(r.created_at for r in rows)
    finished_at = max(r.updated_at for r in rows)
    duration = (
        datetime.fromisoformat(finished_at) - datetime.fromisoformat(started_at)
    ).total_seconds()

    status_counts: dict[str, int] = {}
    failed_files = []
    for r in rows:
        status_counts[r.status] = status_counts.get(r.status, 0) + 1
        if r.status == "failed":
            failed_files.append({"input_path": r.input_path, "error": r.error})

    passthrough_count = skip_text_count = ocr_count = 0
    confidences: list[float] = []
    engine_win_counts: dict[str, int] = {}
    if config.qc_root.exists():
        for qc_path in config.qc_root.rglob("*.qc.json"):
            try:
                kind = _classify_qc(qc_path)
            except (json.JSONDecodeError, OSError):
                continue
            if kind == "passthrough":
                passthrough_count += 1
            elif kind == "skip_text":
                skip_text_count += 1
            else:
                ocr_count += 1
                summary = json.loads(qc_path.read_text(encoding="utf-8"))
                if summary.get("mean_confidence") is not None:
                    confidences.append(summary["mean_confidence"])
                for engine, n in summary.get("engine_win_counts", {}).items():
                    engine_win_counts[engine] = engine_win_counts.get(engine, 0) + n

    mean_confidence = (sum(confidences) / len(confidences)) if confidences else None

    summary_row = {
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_seconds": duration,
        "total_files": len(rows),
        "done_count": status_counts.get("done", 0),
        "failed_count": status_counts.get("failed", 0),
        "flagged_count": status_counts.get("flagged", 0),
        "passthrough_count": passthrough_count,
        "skip_text_count": skip_text_count,
        "ocr_count": ocr_count,
        "mean_confidence": mean_confidence,
        "engine_win_counts": engine_win_counts,
        "failed_files": failed_files,
    }

    with _connect(history_db_path(config)) as conn:
        conn.execute(
            """
            INSERT INTO runs (
                started_at, finished_at, duration_seconds, total_files,
                done_count, failed_count, flagged_count,
                passthrough_count, skip_text_count, ocr_count,
                mean_confidence, engine_win_counts, failed_files
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                summary_row["started_at"],
                summary_row["finished_at"],
                summary_row["duration_seconds"],
                summary_row["total_files"],
                summary_row["done_count"],
                summary_row["failed_count"],
                summary_row["flagged_count"],
                summary_row["passthrough_count"],
                summary_row["skip_text_count"],
                summary_row["ocr_count"],
                summary_row["mean_confidence"],
                json.dumps(summary_row["engine_win_counts"]),
                json.dumps(summary_row["failed_files"]),
            ),
        )
        summary_row["id"] = conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]

    return summary_row


def list_runs(config: Config, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
    with _connect(history_db_path(config)) as conn:
        rows = conn.execute(
            "SELECT * FROM runs ORDER BY id DESC LIMIT ? OFFSET ?", (limit, offset)
        ).fetchall()
        return [_row_to_dict(r) for r in rows]


def get_run(config: Config, run_id: int) -> Optional[dict[str, Any]]:
    with _connect(history_db_path(config)) as conn:
        row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return _row_to_dict(row) if row else None


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    d["engine_win_counts"] = json.loads(d["engine_win_counts"])
    d["failed_files"] = json.loads(d["failed_files"])
    return d
