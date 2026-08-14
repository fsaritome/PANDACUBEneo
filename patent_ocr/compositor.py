"""Sandwich compositor (§5.10): OCRmyPDF does the actual compositing (image
preservation, --skip-text page-level protection, output PDF assembly) — we
only swap in our OCR engine plugin and lock down preprocessing per R6.

The plugin is registered via the `ocrmypdf` entry point in pyproject.toml, so
OCRmyPDF auto-discovers it whenever this package is installed — it must NOT
also be passed via `plugins=[...]` here, or OCRmyPDF raises "Plugin already
registered under a different name" (same module, two different registered
names: the entry point name vs. the module path).
"""
from __future__ import annotations

import os
from pathlib import Path

import ocrmypdf

from patent_ocr.config import Config


def run_ocr_sandwich(input_pdf: Path, output_pdf: Path, config: Config, config_path: str | None) -> None:
    """Run the full OCR-and-sandwich pipeline for one file via OCRmyPDF.

    R6: despeckle/denoise is never wired up here, full stop — not exposed as a
    kwarg at all. Only deskew and contrast normalization are optional
    non-destructive preprocessing steps, both defaulting to disabled per config.
    """
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    # Child processes re-import the plugin fresh; env var is how they find the config.
    if config_path:
        os.environ["PATENT_OCR_CONFIG_PATH"] = str(Path(config_path).resolve())

    # disable_passthrough also overrides OCRmyPDF's own page-level skip-text
    # backstop: force_ocr rasterizes and re-OCRs every page unconditionally,
    # even ones that already carry a text layer. Without this, OCRmyPDF's
    # default skip_text=True would silently leave already-text-bearing pages
    # untouched regardless of our own passthrough.py check.
    force = config.watcher.disable_passthrough
    # GPU-backed engines can't survive OCRmyPDF's own internal page-level
    # multiprocessing (default jobs=cpu_count()): CUDA contexts don't survive
    # fork(), so every forked page-worker re-initializes its own independent
    # copy of the model on the GPU instead of sharing the one already loaded
    # in this process — with a multi-GB VLM that's an OOM waiting to happen.
    # Force single-process (in-thread) page handling whenever the configured
    # primary engine is GPU-backed.
    use_gpu = bool(config.engine.engine_options.get(config.engine.primary, {}).get("use_gpu"))
    extra: dict = {}
    if config.preprocess.oversample_dpi:
        extra["oversample"] = config.preprocess.oversample_dpi
    ocrmypdf.ocr(
        str(input_pdf),
        str(output_pdf),
        language=config.languages,
        deskew=config.preprocess.deskew,
        skip_text=not force,  # page-level backstop: never re-OCR pages that already have sane text
        clean=False,  # R6: destructive cleanup (despeckle-like) never enabled
        clean_final=False,  # R6: never replace the visible final page image
        remove_background=False,
        force_ocr=force,
        jobs=1 if use_gpu else None,
        progress_bar=False,
        **extra,
    )
