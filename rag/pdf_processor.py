import hashlib
import io
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from rag.ocr import OCRDocument


def document_hash(data):
    return hashlib.sha256(data).hexdigest()


def process_pdf(filename, data, settings, progress=None):
    """Extract each page independently; unreadable pages never shift citations."""
    filename = Path(filename.replace("\\", "/")).name
    warnings, chunks = [], []
    try:
        reader = PdfReader(io.BytesIO(data), strict=False)
        if reader.is_encrypted and not reader.decrypt(""):
            return [], {"source": filename, "pages": 0, "text_pages": 0}, [
                f"{filename}: password protected; upload an unlocked copy."]
        pages = len(reader.pages)
    except Exception:
        return [], {"source": filename, "pages": 0, "text_pages": 0}, [
            f"{filename}: invalid or unreadable PDF."]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    text_pages = ocr_pages = 0
    digest = document_hash(data)
    ocr = OCRDocument(data)
    ocr_available = True
    try:
        for number, page in enumerate(reader.pages, 1):
            try:
                text = (page.extract_text() or "").strip()
            except Exception:
                warnings.append(f"{filename}: page {number} could not be read as text.")
                text = ""
            if not text and ocr_available:
                if progress:
                    progress((number - 1) / pages, text=f"{filename}: local OCR on page {number}/{pages} (CPU)")
                try:
                    text = ocr.read_page(number - 1).strip()
                    ocr_pages += bool(text)
                except ImportError:
                    warnings.append(f"{filename}: local OCR dependencies are missing. Run python -m pip install -r requirements.txt.")
                    ocr_available = False
                except Exception:
                    warnings.append(f"{filename}: OCR could not read page {number}; try a clearer, upright scan.")
            if progress:
                progress(number / pages, text=f"{filename}: read page {number}/{pages}")
            if not text:
                continue
            text_pages += 1
            chunks.extend({"source": filename, "page": number, "text": chunk,
                           "document_hash": digest} for chunk in splitter.split_text(text) if chunk.strip())
    finally:
        ocr.close()
    if text_pages < pages:
        warnings.append(f"{filename}: {pages - text_pages} pages have no readable text after local OCR; they may be blank or low-quality scans.")
    if not chunks:
        warnings.append(f"{filename}: no readable text to index.")
    return chunks, {"source": filename, "pages": pages, "text_pages": text_pages,
                    "ocr_pages": ocr_pages, "hash": digest, "chunks": len(chunks)}, warnings
