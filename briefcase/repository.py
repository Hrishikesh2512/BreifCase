from __future__ import annotations

import uuid
from datetime import datetime, timezone

import numpy as np

from briefcase.db import get_connection, transaction
from briefcase.models import Notebook, Source


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# ---------------- Notebooks ----------------


def create_notebook(title: str, description: str = "") -> Notebook:
    nb_id = _new_id("nb")
    created = _now()
    with transaction() as conn:
        conn.execute(
            "INSERT INTO notebooks (id, title, description, created_at) VALUES (?, ?, ?, ?)",
            (nb_id, title, description, created),
        )
    return Notebook(id=nb_id, title=title, description=description, created_at=created)


def list_notebooks() -> list[Notebook]:
    rows = get_connection().execute(
        """
        SELECT n.*, COUNT(s.id) AS source_count
        FROM notebooks n
        LEFT JOIN sources s ON s.notebook_id = n.id
        GROUP BY n.id
        ORDER BY n.created_at DESC
        """
    ).fetchall()
    return [
        Notebook(
            id=r["id"],
            title=r["title"],
            description=r["description"],
            created_at=r["created_at"],
            source_count=r["source_count"],
        )
        for r in rows
    ]


def get_notebook(notebook_id: str) -> Notebook | None:
    r = get_connection().execute(
        """
        SELECT n.*, COUNT(s.id) AS source_count
        FROM notebooks n LEFT JOIN sources s ON s.notebook_id = n.id
        WHERE n.id = ? GROUP BY n.id
        """,
        (notebook_id,),
    ).fetchone()
    if r is None:
        return None
    return Notebook(
        id=r["id"],
        title=r["title"],
        description=r["description"],
        created_at=r["created_at"],
        source_count=r["source_count"],
    )


def delete_notebook(notebook_id: str) -> bool:
    with transaction() as conn:
        cur = conn.execute("DELETE FROM notebooks WHERE id = ?", (notebook_id,))
    return cur.rowcount > 0


# ---------------- Sources & chunks ----------------


def add_source(
    notebook_id: str,
    *,
    title: str,
    source_type: str,
    origin: str,
    full_text: str,
    checksum: str,
    chunks: list[tuple[str, int, int]],  # (text, ordinal, token_count)
    vectors: np.ndarray,
) -> Source:
    source_id = _new_id("src")
    created = _now()
    token_count = sum(tc for _, _, tc in chunks)
    with transaction() as conn:
        conn.execute(
            """INSERT INTO sources
               (id, notebook_id, title, source_type, origin, full_text, token_count, checksum, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (source_id, notebook_id, title, source_type, origin, full_text, token_count, checksum, created),
        )
        for (text, ordinal, tc), vec in zip(chunks, vectors):
            conn.execute(
                """INSERT INTO chunks (id, source_id, notebook_id, ordinal, text, token_count, vector)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (_new_id("chk"), source_id, notebook_id, ordinal, text, tc,
                 np.asarray(vec, dtype=np.float32).tobytes()),
            )
    return Source(
        id=source_id, notebook_id=notebook_id, title=title, source_type=source_type,
        origin=origin, chunk_count=len(chunks), token_count=token_count, created_at=created,
    )


def find_source_by_checksum(notebook_id: str, checksum: str) -> str | None:
    r = get_connection().execute(
        "SELECT id FROM sources WHERE notebook_id = ? AND checksum = ?",
        (notebook_id, checksum),
    ).fetchone()
    return r["id"] if r else None


def list_sources(notebook_id: str) -> list[Source]:
    rows = get_connection().execute(
        """
        SELECT s.*, COUNT(c.id) AS chunk_count
        FROM sources s LEFT JOIN chunks c ON c.source_id = s.id
        WHERE s.notebook_id = ?
        GROUP BY s.id ORDER BY s.created_at ASC
        """,
        (notebook_id,),
    ).fetchall()
    return [
        Source(
            id=r["id"], notebook_id=r["notebook_id"], title=r["title"],
            source_type=r["source_type"], origin=r["origin"], chunk_count=r["chunk_count"],
            token_count=r["token_count"], created_at=r["created_at"],
        )
        for r in rows
    ]


def get_source_text(notebook_id: str, source_id: str) -> tuple[str, str] | None:
    r = get_connection().execute(
        "SELECT title, full_text FROM sources WHERE notebook_id = ? AND id = ?",
        (notebook_id, source_id),
    ).fetchone()
    return (r["title"], r["full_text"]) if r else None


def delete_source(notebook_id: str, source_id: str) -> bool:
    with transaction() as conn:
        cur = conn.execute(
            "DELETE FROM sources WHERE notebook_id = ? AND id = ?", (notebook_id, source_id)
        )
    return cur.rowcount > 0


def load_chunks(notebook_id: str, source_ids: list[str] | None = None) -> list[dict]:
    """Load chunks (with vectors) for retrieval, optionally scoped to sources."""
    conn = get_connection()
    if source_ids:
        placeholders = ",".join("?" for _ in source_ids)
        rows = conn.execute(
            f"""SELECT c.id, c.source_id, c.ordinal, c.text, c.vector, s.title AS source_title
                FROM chunks c JOIN sources s ON s.id = c.source_id
                WHERE c.notebook_id = ? AND c.source_id IN ({placeholders})
                ORDER BY c.source_id, c.ordinal""",
            (notebook_id, *source_ids),
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT c.id, c.source_id, c.ordinal, c.text, c.vector, s.title AS source_title
               FROM chunks c JOIN sources s ON s.id = c.source_id
               WHERE c.notebook_id = ?
               ORDER BY c.source_id, c.ordinal""",
            (notebook_id,),
        ).fetchall()
    return [
        {
            "chunk_id": r["id"],
            "source_id": r["source_id"],
            "source_title": r["source_title"],
            "ordinal": r["ordinal"],
            "text": r["text"],
            "vector": np.frombuffer(r["vector"], dtype=np.float32) if r["vector"] else None,
        }
        for r in rows
    ]
