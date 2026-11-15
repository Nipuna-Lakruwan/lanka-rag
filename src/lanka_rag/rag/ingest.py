"""Load PDFs / text / HTML into LangChain Documents, normalized and chunked."""

from __future__ import annotations

import html
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..text import KUNDDALIYA, detect_language, legacy_font_score, normalize

SUPPORTED = {".pdf", ".txt", ".md", ".html", ".htm"}
LEGACY_THRESHOLD = 0.3

# Paragraphs, lines, then sentence ends (Latin full stop is standard in modern
# Sinhala and Tamil; ෴ appears in older texts), then words.
SEPARATORS = ["\n\n", "\n", f"{KUNDDALIYA} ", ". ", "? ", "! ", "; ", ", ", " ", ""]


@dataclass
class IngestReport:
    files: int = 0
    pages: int = 0
    chunks: int = 0
    skipped_legacy: list[str] = field(default_factory=list)
    skipped_empty: list[str] = field(default_factory=list)
    languages: dict[str, int] = field(default_factory=dict)


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "nav", "footer", "header"}:
            self._skip += 1
        elif tag in {"p", "br", "div", "li", "h1", "h2", "h3", "h4", "tr"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style", "nav", "footer", "header"} and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def _html_to_text(raw: str) -> str:
    p = _TextExtractor()
    p.feed(raw)
    return html.unescape(re.sub(r"[ \t]*\n[ \t]*", "\n", "".join(p.parts)))


def load_pages(path: Path) -> Iterable[tuple[int | None, str]]:
    """Yield (page number or None, raw text) for one file."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        for i, page in enumerate(reader.pages, start=1):
            yield i, page.extract_text() or ""
    elif suffix in {".html", ".htm"}:
        yield None, _html_to_text(path.read_text(encoding="utf-8", errors="replace"))
    else:
        yield None, path.read_text(encoding="utf-8", errors="replace")


def iter_files(paths: Iterable[Path]) -> Iterable[Path]:
    for p in paths:
        if p.is_dir():
            yield from sorted(f for f in p.rglob("*") if f.suffix.lower() in SUPPORTED)
        elif p.suffix.lower() in SUPPORTED:
            yield p


def load_documents(
    paths: Iterable[Path], chunk_size: int = 800, chunk_overlap: int = 120
) -> tuple[list[Document], IngestReport]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=SEPARATORS
    )
    report = IngestReport()
    chunks: list[Document] = []
    for path in iter_files(paths):
        report.files += 1
        for page, raw in load_pages(path):
            report.pages += 1
            where = f"{path.name}" + (f" p.{page}" if page else "")
            text = normalize(raw)
            if len(text) < 20:
                report.skipped_empty.append(where)
                continue
            if legacy_font_score(text) >= LEGACY_THRESHOLD:
                # Indexing this would put Latin garbage into the vector store.
                report.skipped_legacy.append(where)
                continue
            lang = detect_language(text)
            meta = {"source": str(path), "page": page, "lang": lang}
            for chunk in splitter.split_text(text):
                chunks.append(Document(page_content=chunk, metadata=dict(meta)))
                report.languages[lang] = report.languages.get(lang, 0) + 1
    report.chunks = len(chunks)
    return chunks, report
