"""Worker pool consuming the file queue (§5.1). GPU mode uses threads (shared
CUDA context, no fork-induced segfaults); CPU mode uses processes for true
parallelism around the GIL.
"""
from __future__ import annotations

import logging
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

logger = logging.getLogger(__name__)


def _is_gpu_mode(config_path: str | None) -> bool:
    from patent_ocr.config import load_config
    cfg = load_config(config_path)
    # Check whichever engine is actually configured as primary, not a hardcoded
    # name — every GPU-backed engine (paddleocr, paddleocr_vl, surya, ...) needs
    # the shared-CUDA-context ThreadPoolExecutor below, not just "paddleocr".
    return bool(cfg.engine.engine_options.get(cfg.engine.primary, {}).get("use_gpu"))


def _worker_task(input_file_str: str, config_path: str | None, force: bool = True, worker_index: int = 0) -> None:
    from patent_ocr.config import load_config
    from patent_ocr.file_processor import process_file
    from patent_ocr.ledger import Ledger
    from patent_ocr.logging_setup import setup_logging

    config = load_config(config_path)
    setup_logging(config.log_level)
    ledger = Ledger(config.ledger_path)
    process_file(Path(input_file_str), config, config_path, ledger, force=force)


class WorkerPool:
    def __init__(self, max_workers: int, config_path: str | None):
        self.config_path = config_path
        # GPU: use threads — PaddleOCR releases GIL during inference, so threads
        # give real parallelism and share one CUDA context (no fork segfaults).
        # CPU: use processes to bypass the GIL for CPU-bound OCR work.
        if _is_gpu_mode(config_path):
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
        else:
            self.executor = ProcessPoolExecutor(max_workers=max_workers)
        self._in_flight: set[str] = set()
        self._lock = threading.Lock()

    def submit(self, input_file: Path, force: bool = True) -> None:
        key = str(input_file.resolve())
        with self._lock:
            if key in self._in_flight:
                return
            self._in_flight.add(key)

        with self._lock:
            worker_index = len(self._in_flight)
        future = self.executor.submit(_worker_task, key, self.config_path, force, worker_index)

        def _done(_fut, key=key):
            with self._lock:
                self._in_flight.discard(key)
            exc = _fut.exception()
            if exc:
                logger.error("worker task failed for %s: %s", key, exc)

        future.add_done_callback(_done)

    def shutdown(self, wait: bool = True) -> None:
        self.executor.shutdown(wait=wait)
