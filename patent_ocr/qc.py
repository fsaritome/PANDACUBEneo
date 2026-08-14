"""QC / observability (§5.11): per-page QC written during OCR, aggregated into
one sidecar JSON per output file, plus a flagged-files report for the CLI.
"""
from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path
from typing import Any


def page_sort_key(source_name: str) -> str:
    """Zero-padded ordering prefix derived from OCRmyPDF's page filenames.

    OCRmyPDF hands the plugin rasters named like `000003_rasterize.png`, so the
    leading digits are the page sequence. Without this the per-page files were
    named by uuid4 and sorted randomly, making the aggregated `pages` list -
    and every index in `flagged_pages` - meaningless.
    """
    match = re.search(r"\d+", source_name or "")
    return match.group(0).zfill(8) if match else "99999999"


def write_page_qc(qc: dict, source_name: str = "") -> None:
    """Called from the OCRmyPDF plugin (possibly in a worker process) to persist
    one page's QC data. Looked up later by `aggregate_qc_dir` in the parent process."""
    qc_dir = os.environ.get("PATENT_OCR_QC_DIR")
    if not qc_dir:
        return
    path = Path(qc_dir)
    path.mkdir(parents=True, exist_ok=True)
    name = f"{page_sort_key(source_name)}_{uuid.uuid4().hex}.json"
    (path / name).write_text(json.dumps(qc), encoding="utf-8")


def aggregate_qc_dir(qc_dir: Path) -> dict[str, Any]:
    """Combine all per-page QC files written during one file's OCR run into a
    single file-level summary."""
    pages = []
    if qc_dir.exists():
        for p in sorted(qc_dir.glob("*.json")):
            pages.append(json.loads(p.read_text(encoding="utf-8")))

    languages: set[str] = set()
    layout_types: set[str] = set()
    engines_used: set[str] = set()
    engine_win_counts: dict[str, int] = {}
    flagged_pages = []
    fallback_fired = False
    confidences = []

    for i, page in enumerate(pages):
        languages.update(page.get("languages", []))
        if page.get("layout_type"):
            layout_types.add(page["layout_type"])
        for region in page.get("regions", []):
            engines_used.update(region.get("engines_used", []))
            winner = region.get("winner")
            if winner:
                engine_win_counts[winner] = engine_win_counts.get(winner, 0) + 1
            conf = region.get("confidence", {}).get("mean")
            if conf is not None:
                confidences.append(conf)
        if page.get("flagged"):
            flagged_pages.append(i)
        fallback_fired = fallback_fired or page.get("fallback_fired", False)

    return {
        "pages": pages,
        "languages": sorted(languages),
        "layout_types": sorted(layout_types),
        "engines_used": sorted(engines_used),
        "engine_win_counts": engine_win_counts,
        "flagged_pages": flagged_pages,
        "flagged": bool(flagged_pages),
        "fallback_fired": fallback_fired,
        "mean_confidence": (sum(confidences) / len(confidences)) if confidences else None,
    }


def write_sidecar(sidecar_path: Path, summary: dict[str, Any]) -> Path:
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return sidecar_path


def print_flagged_report(ledger) -> None:
    records = ledger.flagged_report()
    if not records:
        print("No flagged files.")
        return
    for rec in records:
        print(f"{rec.input_path}")
        print(f"  output: {rec.output_path}")
        print(f"  reason: {rec.flag_reason}")
        print(f"  languages: {rec.languages}  layout: {rec.layout_type}")
        print(f"  updated: {rec.updated_at}")
        print()
