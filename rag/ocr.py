"""Local CPU OCR for PDF pages without a text layer."""
from functools import lru_cache
from threading import RLock

import numpy as np

# PDFium is not thread-safe; serialize rendering and the shared OCR engine.
_LOCK = RLock()


@lru_cache(maxsize=1)
def _engine():
    from rapidocr_onnxruntime import RapidOCR

    # The pinned package bundles its models and defaults to CPU for all stages.
    return RapidOCR()


class OCRDocument:
    def __init__(self, data):
        self.data = data
        self.pdf = None

    def read_page(self, index):
        with _LOCK:
            import pypdfium2 as pdfium

            if self.pdf is None:
                self.pdf = pdfium.PdfDocument(self.data)
            page = self.pdf[index]
            try:
                # Bound memory for oversized scans; render ordinary pages at 216 dpi.
                scale = min(3.0, 2800 / max(page.get_size()))
                bitmap = page.render(scale=scale)
                try:
                    image = bitmap.to_pil().convert("RGB")
                    try:
                        with image.convert("L") as grayscale:
                            extrema = grayscale.getextrema()
                        if extrema[1] - extrema[0] < 2:
                            return ""
                        # RapidOCR expects BGR arrays, as used by OpenCV.
                        pixels = np.asarray(image)[:, :, ::-1].copy()
                    finally:
                        image.close()
                finally:
                    bitmap.close()
            finally:
                page.close()
            result, _ = _engine()(pixels)
            return "\n".join(item[1] for item in (result or []) if item[1].strip())

    def close(self):
        with _LOCK:
            if self.pdf is not None:
                self.pdf.close()
                self.pdf = None
