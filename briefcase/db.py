from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager

from briefcase.config import get_settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS notebooks (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sources (
    id          TEXT PRIMARY KEY,
    notebook_id TEXT NOT NULL REFERENCES notebooks(id) ON DELETE CASCADE,
    title       TEXT NOT NULL,
    source_type TEXT NOT NULL,
    origin      TEXT NOT NULL DEFAULT '',
    full_text   TEXT NOT NULL DEFAULT '',
    token_count INTEGER NOT NULL DEFAULT 0,
    checksum    TEXT NOT NULL DEFAULT '',
    enabled     INTEGER NOT NULL DEFAULT 1,
    is_web      INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id          TEXT PRIMARY KEY,
    source_id   TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    notebook_id TEXT NOT NULL,
    ordinal     INTEGER NOT NULL,
    text        TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    vector      BLOB
);

CREATE TABLE IF NOT EXISTS drafts (
    id          TEXT PRIMARY KEY,
    notebook_id TEXT NOT NULL UNIQUE REFERENCES notebooks(id) ON DELETE CASCADE,
    title       TEXT NOT NULL DEFAULT 'Untitled draft',
    body        TEXT NOT NULL DEFAULT '',
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS draft_versions (
    id         TEXT PRIMARY KEY,
    draft_id   TEXT NOT NULL REFERENCES drafts(id) ON DELETE CASCADE,
    body       TEXT NOT NULL,
    note       TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sources_notebook ON sources(notebook_id);
CREATE INDEX IF NOT EXISTS idx_chunks_notebook ON chunks(notebook_id);
CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_id);
CREATE INDEX IF NOT EXISTS idx_draft_versions_draft ON draft_versions(draft_id);
"""

_local = threading.local()


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(get_settings().db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def get_connection() -> sqlite3.Connection:
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = _connect()
        _local.conn = conn
    return conn


@contextmanager
def transaction() -> Iterator[sqlite3.Connection]:
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def _migrate(conn: sqlite3.Connection) -> None:
    """Lightweight, additive migrations for existing databases."""
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(sources)").fetchall()}
    if "enabled" not in cols:
        conn.execute("ALTER TABLE sources ADD COLUMN enabled INTEGER NOT NULL DEFAULT 1")
    if "is_web" not in cols:
        conn.execute("ALTER TABLE sources ADD COLUMN is_web INTEGER NOT NULL DEFAULT 0")


def init_db() -> None:
    conn = get_connection()
    conn.executescript(_SCHEMA)
    _migrate(conn)
    conn.commit()
