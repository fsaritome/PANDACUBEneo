"""Worker pool consuming the file queue (§5.1). Process-based so heavy OCR
engines (esp. GPU-backed) get real parallelism without one crashed worker
taking down the whole pipeline.
"""
from __future__ import annotations

import logging
import threading
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

logger = logging.getLogger(__name__)


def _worker_task(input_file_str: str, config_path: str | None, force: bool = True) -> None:
    # Re-imported fresh in the worker process — cheap modules only at import time,
    # heavy OCR engines are lazily instantiated on first use and cached per worker.
    from patent_ocr.config import load_config
    from patent_ocr.file_processor import process_file
    from patent_ocr.ledger import Ledger
    from patent_ocr.logging_setup import setup_logging

    config = load_config(config_path)
    # Windows uses spawn: child processes don't inherit the parent's logging
    # config, so without this, per-file success/failure logs are silently lost.
    setup_logging(config.log_level)
    ledger = Ledger(config.ledger_path)
    process_file(Path(input_file_str), config, config_path, ledger, force=force)


class WorkerPool:
    def __init__(self, max_workers: int, config_path: str | None):
        self.config_path = config_path
        self.executor = ProcessPoolExecutor(max_workers=max_workers)
        self._in_flight: set[str] = set()
        self._lock = threading.Lock()

    def submit(self, input_file: Path, force: bool = True) -> None:
        key = str(input_file.resolve())
        with self._lock:
            if key in self._in_flight:
                return
            self._in_flight.add(key)

        future = self.executor.submit(_worker_task, key, self.config_path, force)

        def _done(_fut, key=key):
            with self._lock:
                self._in_flight.discard(key)
            exc = _fut.exception()
            if exc:
                logger.error("worker task failed for %s: %s", key, exc)

        future.add_done_callback(_done)

    def shutdown(self, wait: bool = True) -> None:
        self.executor.shutdown(wait=wait)
