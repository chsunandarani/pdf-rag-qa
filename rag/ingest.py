"""Load PDFs and split them into overlapping chunks that remember where they came from."""
from dataclasses import dataclass
from pathlib import Path
import re

from pypdf import PdfReader


@dataclass
class Chunk:
    id: int
    text: str
    source: str  # file name
    page: int    # 1-based page number


def _clean(text: str) -> str:
    text = text.replace("­", "")                # soft hyphens
    text = re.sub(r"-\n(\w)", r"\1", text)           # words split across lines
    return re.sub(r"\s+", " ", text).strip()


def load_pdfs(folder: str | Path) -> list[tuple[str, int, str]]:
    """Return (file name, page number, page text) for every page of every PDF in a folder."""
    pages = []
    for path in sorted(Path(folder).glob("*.pdf")):
        reader = PdfReader(path)
        for i, page in enumerate(reader.pages, start=1):
            text = _clean(page.extract_text() or "")
            if text:
                pages.append((path.name, i, text))
    if not pages:
        raise FileNotFoundError(f"No readable PDFs found in {folder}")
    return pages


def chunk_pages(pages, chunk_words: int = 80, overlap: int = 20) -> list[Chunk]:
    """Split each page into windows of `chunk_words` words, overlapping by `overlap` words.

    Chunks never cross a page boundary, so every chunk can be cited with an exact page.
    """
    if overlap >= chunk_words:
        raise ValueError("overlap must be smaller than chunk_words")
    chunks, step = [], chunk_words - overlap
    for source, page_no, text in pages:
        words = text.split()
        for start in range(0, max(len(words) - overlap, 1), step):
            piece = " ".join(words[start:start + chunk_words])
            chunks.append(Chunk(len(chunks), piece, source, page_no))
    return chunks
