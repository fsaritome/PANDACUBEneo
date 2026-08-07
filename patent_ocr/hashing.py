"""Streaming content hashing used by the ledger for idempotency checks."""
from __future__ import annotations

import hashlib
from pathlib import Path

_CHUNK_SIZE = 1024 * 1024


def hash_file(path: str | Path) -> str:
    """Return the sha256 hex digest of a file's contents, read in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()
