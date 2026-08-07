"""R3: output tree mirrors input tree, path-for-path, no flattening/renaming."""
from __future__ import annotations

from pathlib import Path


def output_path_for(input_file: Path, input_root: Path, output_root: Path) -> Path:
    """Compute the mirrored output path for an input file.

    output_root / relative_path(input_file, input_root), preserving the
    original filename and directory structure exactly.
    """
    rel = input_file.resolve().relative_to(input_root.resolve())
    return (output_root / rel).resolve()


def qc_path_for(input_file: Path, input_root: Path, qc_root: Path) -> Path:
    """Mirrored QC sidecar path, kept in a separate tree from output_root so
    the output folder only ever contains the OCR'd PDFs themselves."""
    rel = input_file.resolve().relative_to(input_root.resolve())
    return (qc_root / rel).with_suffix(rel.suffix + ".qc.json").resolve()


def failed_path_for(input_file: Path, input_root: Path, failed_root: Path) -> Path:
    """Mirrored path under failed_root, same relative path/filename as the input
    — where a file is moved to (out of input_root) when processing raises."""
    rel = input_file.resolve().relative_to(input_root.resolve())
    return (failed_root / rel).resolve()


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def iter_input_files(input_root: Path, extensions: list[str]) -> "list[Path]":
    """Recursively enumerate every file under input_root matching extensions."""
    exts = {e.lower() for e in extensions}
    return [
        p
        for p in input_root.rglob("*")
        if p.is_file() and p.suffix.lower() in exts
    ]
