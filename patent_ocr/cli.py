"""CLI entry point: `patent-ocr run|sweep|report-flagged|status`."""
from __future__ import annotations

import argparse
import logging
import shutil
import signal
import sys
import time

from patent_ocr.config import load_config
from patent_ocr.history import record_run
from patent_ocr.ledger import Ledger
from patent_ocr.logging_setup import setup_logging
from patent_ocr.pipeline import WorkerPool
from patent_ocr.qc import print_flagged_report
from patent_ocr.watcher import backlog_sweep, start_watcher

log = logging.getLogger(__name__)


def _startup_reconcile(config) -> None:
    """Clear state a previously killed run could not clean up itself."""
    stale = Ledger(config.ledger_path).reconcile_interrupted()
    if stale:
        log.warning("reconciled %d interrupted file(s) left as 'processing'", stale)

    work_dir = config.work_dir
    orphans = [d for d in work_dir.iterdir() if d.is_dir()] if work_dir.exists() else []
    for orphan in orphans:
        shutil.rmtree(orphan, ignore_errors=True)
    if orphans:
        log.warning("removed %d orphaned work dir(s)", len(orphans))


def _cmd_run(args) -> None:
    config = load_config(args.config)
    setup_logging(config.log_level)
    _startup_reconcile(config)
    pool = WorkerPool(config.watcher.max_workers, args.config)

    observer = start_watcher(config, pool)  # live first, per R2
    backlog_sweep(config, pool, force=not args.skip_done)

    stop = {"flag": False}

    def _handle_sigint(signum, frame):
        stop["flag"] = True

    signal.signal(signal.SIGINT, _handle_sigint)
    signal.signal(signal.SIGTERM, _handle_sigint)

    try:
        while not stop["flag"]:
            time.sleep(1)
    finally:
        observer.stop()
        observer.join()
        pool.shutdown(wait=True)


def _cmd_sweep(args) -> None:
    config = load_config(args.config)
    setup_logging(config.log_level)
    _startup_reconcile(config)
    pool = WorkerPool(config.watcher.max_workers, args.config)
    backlog_sweep(config, pool, force=not args.skip_done)
    pool.shutdown(wait=True)
    record_run(config)


def _cmd_report_flagged(args) -> None:
    config = load_config(args.config)
    ledger = Ledger(config.ledger_path)
    print_flagged_report(ledger)


def _cmd_status(args) -> None:
    config = load_config(args.config)
    ledger = Ledger(config.ledger_path)
    for status, count in sorted(ledger.status_counts().items()):
        print(f"{status:12s} {count}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="patent-ocr")
    parser.add_argument("--config", "-c", default=None, help="Path to config YAML")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("run", help="Start the watcher + backlog sweep + worker pool (foreground)").set_defaults(func=_cmd_run)
    sub.add_parser("sweep", help="Run a one-off backlog sweep and exit when the queue drains").set_defaults(func=_cmd_sweep)
    sub.add_parser("report-flagged", help="List files flagged for human review").set_defaults(func=_cmd_report_flagged)
    sub.add_parser("status", help="Show ledger status counts").set_defaults(func=_cmd_status)

    for name in ("run", "sweep"):
        sub.choices[name].add_argument(
            "--skip-done", action="store_true",
            help="Skip files whose content hash already has a done/flagged ledger record "
                 "(opt-in old idempotent behavior; default is to always re-OCR everything in input/)",
        )

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
