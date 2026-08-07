"""Configuration loading for the patent OCR pipeline.

Single source of truth for all tunables. Loaded from a YAML file with
sane defaults so the pipeline can run with just `input_root`/`output_root` set.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class EngineConfig:
    # Which engines are active, in priority order. First is "primary".
    primary: str = "tesseract"
    # Optional second engine used per §5.7 strategy below.
    secondary: str | None = None
    # "single" | "always_parallel" | "low_confidence_only"
    strategy: str = "single"
    # Per-word confidence (0-100) below which a region is considered low-confidence.
    low_confidence_word_threshold: float = 60.0
    # Fraction of low-confidence words in a region that triggers fallback/secondary.
    low_confidence_region_fraction: float = 0.15
    # Per-engine constructor kwargs, keyed by engine name (e.g. {"paddleocr": {"use_gpu": false}}).
    engine_options: dict[str, dict] = field(default_factory=dict)


@dataclass
class FallbackConfig:
    enabled: bool = False
    # OpenAI-compatible chat endpoint serving a local vision model (e.g. ai01 vLLM/Ollama).
    base_url: str = "http://ai01:11434/v1"
    api_key: str = "not-needed"
    model: str = "qwen2-vl"
    # (b) per spec 5.8: never silently trust the LLM text as the layer; only flag for review
    # unless this is explicitly toggled on after validation.
    apply_as_text_layer: bool = False
    timeout_seconds: int = 120


@dataclass
class LayoutConfig:
    # Minimum whitespace gap width (as a fraction of page width) to count as a column gap.
    min_gap_fraction: float = 0.015
    # Maximum width (as a fraction of page width) for a leading band to be classified as
    # a margin line-number strip rather than a body column.
    margin_max_width_fraction: float = 0.12
    # Minimum ink-density ratio (relative to page peak) below which a column is "empty" (gap).
    gap_density_threshold: float = 0.02


@dataclass
class PreprocessConfig:
    # R6: destructive preprocessing (despeckle/denoise) is never available, not just off-by-default.
    # Only these two non-destructive operations are configurable, and both default to disabled.
    deskew: bool = False
    contrast_normalize: bool = False


@dataclass
class WatcherConfig:
    input_root: str = "./input"
    output_root: str = "./output"
    qc_root: str = "./qc"
    failed_root: str = "./failed"
    file_extensions: list[str] = field(default_factory=lambda: [".pdf"])
    max_workers: int = 4
    ledger_path: str = "./state/ledger.sqlite3"
    work_dir: str = "./state/work"


@dataclass
class Config:
    watcher: WatcherConfig = field(default_factory=WatcherConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    fallback: FallbackConfig = field(default_factory=FallbackConfig)
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    languages: list[str] = field(default_factory=lambda: ["eng", "deu", "fra"])
    log_level: str = "INFO"

    @property
    def input_root(self) -> Path:
        return Path(self.watcher.input_root).resolve()

    @property
    def output_root(self) -> Path:
        return Path(self.watcher.output_root).resolve()

    @property
    def qc_root(self) -> Path:
        return Path(self.watcher.qc_root).resolve()

    @property
    def failed_root(self) -> Path:
        return Path(self.watcher.failed_root).resolve()

    @property
    def ledger_path(self) -> Path:
        return Path(self.watcher.ledger_path).resolve()

    @property
    def work_dir(self) -> Path:
        return Path(self.watcher.work_dir).resolve()


def _merge(base: Any, override: dict) -> Any:
    for key, value in override.items():
        if not hasattr(base, key):
            raise ValueError(f"Unknown config key: {key}")
        current = getattr(base, key)
        if dataclasses.is_dataclass(current) and isinstance(value, dict):
            _merge(current, value)
        else:
            setattr(base, key, value)
    return base


def load_config(path: str | Path | None) -> Config:
    cfg = Config()
    if path is None:
        return cfg
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return _merge(cfg, raw)
