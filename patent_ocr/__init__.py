"""Patent OCR pipeline: self-hosted hot-folder OCR-and-sandwich for patent PDFs."""

from PIL import Image

# Patent scans are large, trusted, locally-supplied documents, not
# untrusted web uploads — disable Pillow's decompression-bomb guard
# (default ~178M px) so legitimately large/high-DPI pages don't fail
# with DecompressionBombError during rasterization/OCR.
Image.MAX_IMAGE_PIXELS = None

__version__ = "0.1.0"
