import hashlib
import io
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


def document_hash(data):
    return hashlib.sha256(data).hexdigest()


def process_pdf(filename, data, settings):
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
    text_pages = 0
    digest = document_hash(data)
    for number, page in enumerate(reader.pages, 1):
        try:
            text = (page.extract_text() or "").strip()
        except Exception:
            warnings.append(f"{filename}: page {number} could not be read.")
            continue
        if not text:
            continue
        text_pages += 1
        chunks.extend({"source": filename, "page": number, "text": chunk,
                       "document_hash": digest} for chunk in splitter.split_text(text) if chunk.strip())
    if text_pages < pages:
        warnings.append(f"{filename}: {pages - text_pages} pages have no extractable text; scanned pages need OCR.")
    if not chunks:
        warnings.append(f"{filename}: no readable text to index.")
    return chunks, {"source": filename, "pages": pages, "text_pages": text_pages,
                    "hash": digest, "chunks": len(chunks)}, warnings
