import io
import json

import numpy as np
import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject


def make_pdf(texts, password=None):
    writer = PdfWriter()
    for text in texts:
        page = writer.add_blank_page(width=600, height=800)
        font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                                 NameObject("/Subtype"): NameObject("/Type1"),
                                 NameObject("/BaseFont"): NameObject("/Helvetica")})
        page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})})
        if text:
            stream = DecodedStreamObject()
            escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            stream.set_data(f"BT /F1 12 Tf 50 700 Td ({escaped}) Tj ET".encode())
            page[NameObject("/Contents")] = writer._add_object(stream)
    if password:
        writer.encrypt(password)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


class FakeService:
    """Deterministic topic vectors; real FAISS and PDF extraction run in tests."""
    def __init__(self, dimension=768):
        self.dimension = dimension
        self.calls = []
        self.prompts = []

    def embed(self, texts, task_type):
        self.calls.append((list(texts), task_type))
        result = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for i, text in enumerate(texts):
            result[i, 0 if "mapreduce" in text.lower() else 1] = 1
        return result

    def generate(self, prompt, instruction):
        self.prompts.append((json.loads(prompt), instruction))
        payload = json.loads(prompt)
        if "history" in payload:
            return payload["question"].replace("its", "MapReduce's").replace("it", "MapReduce")
        return "MapReduce processes data in parallel. " + payload["passages"][0]["citation"]


@pytest.fixture
def service():
    return FakeService()


@pytest.fixture
def documents():
    return [("lecture.pdf", make_pdf(["MapReduce processes data in parallel.", "", "MapReduce advantages include scalability."])),
            ("biology.pdf", make_pdf(["Photosynthesis uses sunlight in plants."]))]
