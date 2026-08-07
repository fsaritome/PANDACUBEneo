"""Common OCR engine interface (§5.7). Engines are swappable, never hard-coded."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

# left, top, right, bottom in pixel coordinates of whatever image was passed in.
BBox = tuple[int, int, int, int]


@dataclass
class Word:
    text: str
    bbox: BBox
    confidence: float  # 0-100
    engine: str = ""  # which engine produced this word, for QC/reconciliation


@dataclass
class OcrResult:
    """Full recognition result for one region image."""
    words: list[Word] = field(default_factory=list)

    @property
    def mean_confidence(self) -> float:
        if not self.words:
            return 0.0
        return sum(w.confidence for w in self.words) / len(self.words)

    def low_confidence_fraction(self, threshold: float) -> float:
        if not self.words:
            return 1.0
        low = sum(1 for w in self.words if w.confidence < threshold)
        return low / len(self.words)


class OCREngine(ABC):
    """Pluggable OCR engine. Implementations must be swappable per §5.7."""

    name: str = "unnamed"

    @abstractmethod
    def recognize(self, region_image, lang_hint: list[str] | None = None) -> list[Word]:
        """Run OCR on a single region image (numpy array or PIL.Image).

        Returns word-level results with bboxes in the region image's own pixel
        coordinate space (the caller is responsible for remapping to page
        coordinates during reassembly).
        """
        raise NotImplementedError
