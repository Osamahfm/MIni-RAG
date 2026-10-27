import io
from unittest.mock import patch

import pytest
from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfReader, PdfWriter

from conftest import make_pdf
from rag.config import Settings
from rag.ocr import _engine
from rag.pdf_processor import process_pdf
from rag.vector_store import VectorStore


def scanned_pdf():
    image = Image.new("RGB", (1200, 400), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=44)
    draw.text((60, 60), "MPI uses message passing.", fill="black", font=font)
    draw.text((60, 160), "Processes exchange data.", fill="black", font=font)
    output = io.BytesIO()
    image.save(output, format="PDF", resolution=144)
    image.close()
    return output.getvalue()


@pytest.fixture(scope="module")
def scan():
    return scanned_pdf()


def test_real_local_cpu_ocr_and_original_page_numbers(scan):
    assert not PdfReader(io.BytesIO(scan)).pages[0].extract_text()
    writer = PdfWriter()
    writer.append(io.BytesIO(make_pdf(["Native text stays unchanged.", ""])))
    writer.append(io.BytesIO(scan))
    mixed = io.BytesIO()
    writer.write(mixed)

    chunks, stats, warnings = process_pdf("scanned.pdf", mixed.getvalue(), Settings())
    assert stats["pages"] == 3 and stats["text_pages"] == 2 and stats["ocr_pages"] == 1
    assert chunks[0]["text"] == "Native text stays unchanged."
    assert {c["page"] for c in chunks} == {1, 3}
    assert "message passing" in " ".join(c["text"] for c in chunks if c["page"] == 3).lower()
    engine = _engine()
    sessions = [engine.text_detector.infer.session, engine.text_recognizer.session.session,
                engine.text_cls.infer.session]
    assert all(session.get_providers() == ["CPUExecutionProvider"] for session in sessions)
    assert not any("OCR could not" in warning for warning in warnings)


def test_native_text_never_runs_ocr():
    with patch("rag.pdf_processor.OCRDocument.read_page") as read:
        chunks, stats, _ = process_pdf("text.pdf", make_pdf(["Native PDF text."]), Settings())
    read.assert_not_called()
    assert chunks and stats["ocr_pages"] == 0


def test_ocr_failure_preserves_other_pages():
    with patch("rag.pdf_processor.OCRDocument.read_page", side_effect=RuntimeError("bad scan")):
        chunks, stats, warnings = process_pdf("mixed.pdf", make_pdf(["", "Readable evidence."]), Settings())
    assert {c["page"] for c in chunks} == {2}
    assert stats["ocr_pages"] == 0
    assert any("OCR could not read page 1" in warning for warning in warnings)


def test_missing_ocr_dependency_is_reported_once():
    with patch("rag.pdf_processor.OCRDocument.read_page", side_effect=ImportError) as read:
        chunks, _, warnings = process_pdf("scan.pdf", make_pdf(["", ""]), Settings())
    assert not chunks and read.call_count == 1
    assert sum("dependencies are missing" in warning for warning in warnings) == 1


def test_ocr_text_and_page_metadata_are_cached(tmp_path, service):
    data = make_pdf(["", "Native text."])
    with patch("rag.pdf_processor.OCRDocument.read_page", return_value="MPI uses message passing.") as read:
        db = VectorStore(tmp_path, Settings()).build([("lecture.pdf", data)], service)
    assert read.call_count == 1 and db.documents[0]["ocr_pages"] == 1
    with patch("rag.pdf_processor.OCRDocument.read_page", side_effect=AssertionError("OCR must be cached")):
        cached = VectorStore(tmp_path, Settings()).build([("lecture.pdf", data)], service)
    assert cached.cached and cached.documents[0]["ocr_pages"] == 1
    assert {c["page"] for c in cached.chunks} == {1, 2}
