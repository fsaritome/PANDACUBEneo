"""Hot folder watcher + backlog sweep (§5.1, R1, R2).

The watcher goes live first so no filesystem events are missed, then the
backlog sweep runs (enumerating the whole input tree) — this satisfies R2's
"before or in parallel with going live" requirement without a race where a
file created mid-sweep gets missed by both mechanisms.
"""
from __future__ import annotations

import logging
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from patent_ocr.config import Config
from patent_ocr.pipeline import WorkerPool
from patent_ocr.tree_utils import iter_input_files

logger = logging.getLogger(__name__)


class _Handler(FileSystemEventHandler):
    def __init__(self, config: Config, pool: WorkerPool):
        self.config = config
        self.pool = pool

    def _maybe_submit(self, path_str: str) -> None:
        path = Path(path_str)
        if path.is_file() and path.suffix.lower() in {e.lower() for e in self.config.watcher.file_extensions}:
            # A live drop is an explicit action — always (re-)OCR it, never
            # silently skip due to a prior ledger record (unlike the startup
            # backlog sweep, which stays idempotent for large archives).
            self.pool.submit(path, force=True)

    def on_created(self, event):
        if not event.is_directory:
            self._maybe_submit(event.src_path)

    def on_modified(self, event):
        if not event.is_directory:
            self._maybe_submit(event.src_path)

    def on_moved(self, event):
        if not event.is_directory:
            self._maybe_submit(event.dest_path)


def start_watcher(config: Config, pool: WorkerPool) -> Observer:
    config.input_root.mkdir(parents=True, exist_ok=True)
    handler = _Handler(config, pool)
    observer = Observer()
    observer.schedule(handler, str(config.input_root), recursive=True)
    observer.start()
    logger.info("watcher live on %s", config.input_root)
    return observer


def backlog_sweep(config: Config, pool: WorkerPool, force: bool = True) -> int:
    """Enumerate the entire input tree and (re-)enqueue every matching file.

    Runs every startup (R2) and always reprocesses by default — a file sitting
    in input/ (whether pre-existing, dropped during downtime, or left over
    from a crash) is always OCR'd; pass force=False to restore the old
    idempotent skip-if-already-done behavior.
    """
    files = iter_input_files(config.input_root, config.watcher.file_extensions)
    for f in files:
        pool.submit(f, force=force)
    logger.info("backlog sweep enqueued %d file(s) from %s", len(files), config.input_root)
    return len(files)
