from __future__ import annotations

import ipaddress
import socket
import urllib.error
import urllib.request
from urllib.parse import urlparse

from briefcase.ingest.chunking import Chunker
from briefcase.ingest.parsers import ParsedDocument, ParserError, parse, parse_html_string
from briefcase.models import Source
from briefcase.rag.embeddings import get_embedder
from briefcase.repository import (
    add_source,
    find_source_by_checksum,
    get_source_meta,
    replace_source_content,
)
from briefcase.text_utils import checksum

_MAX_FETCH_BYTES = 3_000_000
_FETCH_TIMEOUT = 20


class IngestionError(RuntimeError):
    pass


def _host_is_public(host: str) -> bool:
    """True only if every address the host resolves to is a routable public IP.
    Blocks localhost, private ranges, link-local, and other internal targets (SSRF)."""
    try:
        infos = socket.getaddrinfo(host, None)
    except Exception:
        return False
    for info in infos:
        try:
            addr = ipaddress.ip_address(info[4][0])
        except ValueError:
            return False
        if (addr.is_private or addr.is_loopback or addr.is_link_local
                or addr.is_reserved or addr.is_multicast or addr.is_unspecified):
            return False
    return True


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urlparse(newurl)
        if parsed.scheme not in ("http", "https") or not parsed.hostname or not _host_is_public(parsed.hostname):
            raise urllib.error.URLError("unsafe redirect target")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def safe_fetch(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise IngestionError("Only http:// and https:// URLs are allowed.")
    if not parsed.hostname or not _host_is_public(parsed.hostname):
        raise IngestionError("Refusing to fetch a private, local, or unresolvable address.")
    opener = urllib.request.build_opener(_SafeRedirect)
    req = urllib.request.Request(url, headers={"User-Agent": "Briefcase/0.1 (+local)"})
    try:
        with opener.open(req, timeout=_FETCH_TIMEOUT) as resp:
            data = resp.read(_MAX_FETCH_BYTES + 1)
    except IngestionError:
        raise
    except Exception as exc:
        raise IngestionError(f"Could not fetch URL: {exc}") from exc
    return data[:_MAX_FETCH_BYTES].decode("utf-8", errors="replace")


class IngestionService:
    def __init__(self) -> None:
        self._chunker = Chunker()

    @property
    def _embedder(self):
        return get_embedder()

    def _chunk_and_embed(self, text: str):
        raw_chunks = self._chunker.chunk(text)
        if not raw_chunks:
            raise IngestionError("No content could be extracted from this source.")
        vectors = self._embedder.embed([c.text for c in raw_chunks])
        return [(c.text, c.ordinal, c.token_count) for c in raw_chunks], vectors

    def ingest_document(
        self, notebook_id: str, doc: ParsedDocument, *, origin: str, is_web: bool = False
    ) -> Source:
        digest = checksum(doc.text)
        if find_source_by_checksum(notebook_id, digest):
            raise IngestionError("This source already exists in the notebook.")
        chunk_rows, vectors = self._chunk_and_embed(doc.text)
        return add_source(
            notebook_id, title=doc.title, source_type=doc.source_type, origin=origin,
            full_text=doc.text, checksum=digest, chunks=chunk_rows, vectors=vectors, is_web=is_web,
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
        raw = safe_fetch(url)
        try:
            doc = parse_html_string(raw, url=url)
        except ParserError as exc:
            raise IngestionError(str(exc)) from exc
        if title:
            doc.title = title
        return self.ingest_document(notebook_id, doc, origin=url, is_web=True)

    def refresh_source(self, notebook_id: str, source_id: str) -> int:
        """Re-fetch a web source and rebuild its chunks in place. Returns chunk count."""
        meta = get_source_meta(notebook_id, source_id)
        if meta is None:
            raise IngestionError("Source not found.")
        if not meta["is_web"] or not str(meta["origin"]).startswith(("http://", "https://")):
            raise IngestionError("Only website sources can be refreshed.")
        raw = safe_fetch(meta["origin"])
        try:
            doc = parse_html_string(raw, url=meta["origin"])
        except ParserError as exc:
            raise IngestionError(str(exc)) from exc
        chunk_rows, vectors = self._chunk_and_embed(doc.text)
        replace_source_content(
            source_id, full_text=doc.text, checksum=checksum(doc.text),
            chunks=chunk_rows, vectors=vectors,
        )
        return len(chunk_rows)
