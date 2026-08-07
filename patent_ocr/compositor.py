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

    ocrmypdf.ocr(
        str(input_pdf),
        str(output_pdf),
        language=config.languages,
        deskew=config.preprocess.deskew,
        skip_text=True,  # page-level backstop: never re-OCR pages that already have sane text
        clean=False,  # R6: destructive cleanup (despeckle-like) never enabled
        clean_final=False,  # R6: never replace the visible final page image
        remove_background=False,
        force_ocr=False,
        progress_bar=False,
    )
