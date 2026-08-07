"""Per-file orchestration: triage -> passthrough check -> OCR sandwich ->
ledger update -> QC sidecar (§4, §5.4, §5.11). This is what the worker pool
and the CLI's one-off `process` command both call.
"""
from __future__ import annotations

import logging
import os
import shutil
import uuid
from pathlib import Path

from patent_ocr.compositor import run_ocr_sandwich
from patent_ocr.config import Config
from patent_ocr.hashing import hash_file
from patent_ocr.ledger import Ledger
from patent_ocr.passthrough import analyze_text_native
from patent_ocr.qc import aggregate_qc_dir, write_sidecar
from patent_ocr.tree_utils import ensure_parent_dir, failed_path_for, output_path_for, qc_path_for

log = logging.getLogger(__name__)


def process_file(input_file: Path, config: Config, config_path: str | None, ledger: Ledger, force: bool = True) -> None:
    input_file = input_file.resolve()
    output_file = output_path_for(input_file, config.input_root, config.output_root)
    qc_file = qc_path_for(input_file, config.input_root, config.qc_root)
    content_hash = hash_file(input_file)

    if not force and not ledger.needs_processing(str(input_file), content_hash):
        log.info("skip: %s (already done, unchanged content; pass force=True to re-OCR)", input_file)
        return

    ledger.enqueue(str(input_file), content_hash)
    ledger.mark_processing(str(input_file))
    ensure_parent_dir(output_file)

    try:
        fully_native, _ = analyze_text_native(input_file)
        if fully_native:
            # §5.4: already has a sane text layer — copy through, don't touch it.
            shutil.copy2(input_file, output_file)
            ledger.mark_done(
                str(input_file), str(output_file),
                engines_used=[], confidence_summary={}, languages=[],
                layout_type="text_native_passthrough", fallback_fired=False,
            )
            write_sidecar(qc_file, {"passthrough": True})
            log.info("passthrough: %s", input_file)
            input_file.unlink(missing_ok=True)
            return

        qc_dir = config.work_dir / uuid.uuid4().hex
        os.environ["PATENT_OCR_QC_DIR"] = str(qc_dir)
        try:
            run_ocr_sandwich(input_file, output_file, config, config_path)
        finally:
            summary = aggregate_qc_dir(qc_dir)
            shutil.rmtree(qc_dir, ignore_errors=True)

        write_sidecar(qc_file, summary)
        ledger.mark_done(
            str(input_file), str(output_file),
            engines_used=summary["engines_used"],
            confidence_summary={"mean": summary["mean_confidence"]},
            languages=summary["languages"],
            layout_type=",".join(summary["layout_types"]) or None,
            fallback_fired=summary["fallback_fired"],
            flagged=summary["flagged"],
            flag_reason=f"{len(summary['flagged_pages'])} page(s) below confidence threshold" if summary["flagged"] else None,
        )
        log.info("done: %s -> %s (flagged=%s)", input_file, output_file, summary["flagged"])
        input_file.unlink(missing_ok=True)
    except Exception as exc:  # noqa: BLE001 - must not crash the worker pool
        log.exception("failed: %s", input_file)
        ledger.mark_failed(str(input_file), str(exc))
        _move_to_failed(input_file, config)


def _move_to_failed(input_file: Path, config: Config) -> None:
    """Pull a failed file out of the hot folder so it isn't retried on every
    sweep and is easy for a human to find; never let this mask the real error."""
    try:
        failed_file = failed_path_for(input_file, config.input_root, config.failed_root)
        ensure_parent_dir(failed_file)
        if failed_file.exists():
            failed_file.unlink()
        shutil.move(str(input_file), str(failed_file))
    except Exception:
        log.exception("could not move failed file to failed_root: %s", input_file)
