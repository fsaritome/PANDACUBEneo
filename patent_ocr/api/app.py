"""FastAPI app exposing the OCR pipeline's live status, run history, and
failure log for the admin dashboard.

Run with: uvicorn patent_ocr.api.app:app --host 0.0.0.0 --port 8000
No auth — intended for internal/VPN network access only.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from patent_ocr import history
from patent_ocr.config import load_config
from patent_ocr.ledger import FileRecord, Ledger

CONFIG_PATH = os.environ.get("PATENT_OCR_CONFIG")
DASHBOARD_DIST = Path(os.environ.get("PATENT_OCR_DASHBOARD_DIST", "dashboard_dist"))

app = FastAPI(title="Patent OCR Dashboard API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _config():
    return load_config(CONFIG_PATH)


def _ledger() -> Ledger:
    return Ledger(_config().ledger_path)


def _elapsed_seconds(created_at: str) -> float:
    started = datetime.fromisoformat(created_at)
    now = datetime.now(timezone.utc)
    return (now - started).total_seconds()


def _record_to_dict(r: FileRecord) -> dict:
    return {
        "input_path": r.input_path,
        "filename": Path(r.input_path).name,
        "output_path": r.output_path,
        "status": r.status,
        "engines_used": r.engines_used,
        "confidence_summary": r.confidence_summary,
        "languages": r.languages,
        "layout_type": r.layout_type,
        "fallback_fired": r.fallback_fired,
        "flagged": r.flagged,
        "flag_reason": r.flag_reason,
        "error": r.error,
        "created_at": r.created_at,
        "updated_at": r.updated_at,
    }


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/status/live")
def status_live():
    cfg = _config()
    ledger = _ledger()
    counts = ledger.status_counts()
    processing = [
        {
            **_record_to_dict(r),
            "elapsed_seconds": _elapsed_seconds(r.created_at),
        }
        for r in ledger.processing_records()
    ]
    # Count actual files in the input directory so the progress bar shows a
    # stable denominator from the start, not one that grows as workers enqueue.
    input_root = cfg.input_root
    input_total = sum(
        1 for ext in cfg.watcher.file_extensions
        for _ in input_root.glob(f"**/*{ext}")
    ) if input_root.exists() else 0
    ledger_total = sum(counts.values())
    # Use whichever is larger: input dir (before sweep completes) or ledger
    # (after input files have been moved/processed and input dir is empty).
    total = max(input_total, ledger_total)
    return {
        "status_counts": counts,
        "total": total,
        "processing": processing,
    }


@app.get("/api/files")
def list_files(status: Optional[str] = None, limit: int = Query(200, le=2000), offset: int = 0):
    ledger = _ledger()
    records = ledger.all_records()
    if status:
        records = [r for r in records if r.status == status]
    total = len(records)
    page = records[offset : offset + limit]
    return {"total": total, "files": [_record_to_dict(r) for r in page]}


@app.get("/api/files/{filename}")
def get_file(filename: str):
    ledger = _ledger()
    for r in ledger.all_records():
        if Path(r.input_path).name == filename:
            return _record_to_dict(r)
    raise HTTPException(status_code=404, detail="file not found")


@app.get("/api/failures")
def failures():
    ledger = _ledger()
    return {"failures": [_record_to_dict(r) for r in ledger.failed_records()]}


@app.get("/api/qc/current")
def qc_current():
    """Aggregate stats across the current run's qc/ directory (may be a
    partially-completed run if called while a sweep is still in progress)."""
    config = _config()
    passthrough = skip_text = ocr = 0
    confidences: list[float] = []
    engine_win_counts: dict[str, int] = {}
    if config.qc_root.exists():
        for qc_path in config.qc_root.rglob("*.qc.json"):
            try:
                summary = json.loads(qc_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if summary.get("passthrough"):
                passthrough += 1
            elif not summary.get("pages"):
                skip_text += 1
            else:
                ocr += 1
                if summary.get("mean_confidence") is not None:
                    confidences.append(summary["mean_confidence"])
                for engine, n in summary.get("engine_win_counts", {}).items():
                    engine_win_counts[engine] = engine_win_counts.get(engine, 0) + n
    return {
        "passthrough_count": passthrough,
        "skip_text_count": skip_text,
        "ocr_count": ocr,
        "mean_confidence": (sum(confidences) / len(confidences)) if confidences else None,
        "engine_win_counts": engine_win_counts,
    }


@app.get("/api/history/runs")
def history_runs(limit: int = Query(50, le=500), offset: int = 0):
    config = _config()
    return {"runs": history.list_runs(config, limit=limit, offset=offset)}


@app.get("/api/history/runs/{run_id}")
def history_run_detail(run_id: int):
    config = _config()
    run = history.get_run(config, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    return run


if DASHBOARD_DIST.exists():
    app.mount("/", StaticFiles(directory=str(DASHBOARD_DIST), html=True), name="dashboard")
