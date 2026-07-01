from __future__ import annotations

import urllib.request

from briefcase.ingest.chunking import Chunker
from briefcase.ingest.parsers import ParsedDocument, ParserError, parse, parse_html_string
from briefcase.models import Source
from briefcase.rag.embeddings import get_embedder
from briefcase.repository import add_source, find_source_by_checksum
from briefcase.text_utils import checksum


class IngestionError(RuntimeError):
    pass


class IngestionService:
    def __init__(self) -> None:
        self._chunker = Chunker()

    @property
    def _embedder(self):
        return get_embedder()

    def ingest_document(self, notebook_id: str, doc: ParsedDocument, *, origin: str) -> Source:
        digest = checksum(doc.text)
        existing = find_source_by_checksum(notebook_id, digest)
        if existing:
            raise IngestionError("This source already exists in the notebook.")

        raw_chunks = self._chunker.chunk(doc.text)
        if not raw_chunks:
            raise IngestionError("No content could be extracted from this source.")

        vectors = self._embedder.embed([c.text for c in raw_chunks])
        chunk_rows = [(c.text, c.ordinal, c.token_count) for c in raw_chunks]
        return add_source(
            notebook_id,
            title=doc.title,
            source_type=doc.source_type,
            origin=origin,
            full_text=doc.text,
            checksum=digest,
            chunks=chunk_rows,
            vectors=vectors,
        )

    def ingest_file(self, notebook_id: str, payload: bytes, filename: str) -> Source:
        try:
            doc = parse(payload, filename=filename)
        except ParserError as exc:
            raise IngestionError(str(exc)) from exc
        return self.ingest_document(notebook_id, doc, origin=filename)

    def ingest_text(self, notebook_id: str, title: str, text: str) -> Source:
        doc = ParsedDocument(text=text.strip(), title=title, source_type="text")
        return self.ingest_document(notebook_id, doc, origin="pasted text")

    def ingest_url(self, notebook_id: str, url: str, title: str | None = None) -> Source:
        if not url.startswith(("http://", "https://")):
            raise IngestionError("URL must start with http:// or https://")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Briefcase/0.1"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except Exception as exc:
            raise IngestionError(f"Could not fetch URL: {exc}") from exc
        try:
            doc = parse_html_string(raw, url=url)
        except ParserError as exc:
            raise IngestionError(str(exc)) from exc
        if title:
            doc.title = title
        return self.ingest_document(notebook_id, doc, origin=url)
