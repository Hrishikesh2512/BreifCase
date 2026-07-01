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
    is_web: bool = False,
) -> Source:
    source_id = _new_id("src")
    created = _now()
    token_count = sum(tc for _, _, tc in chunks)
    with transaction() as conn:
        conn.execute(
            """INSERT INTO sources
               (id, notebook_id, title, source_type, origin, full_text, token_count, checksum, is_web, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (source_id, notebook_id, title, source_type, origin, full_text, token_count, checksum,
             1 if is_web else 0, created),
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
        enabled=True, is_web=is_web,
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
            enabled=bool(r["enabled"]), is_web=bool(r["is_web"]),
        )
        for r in rows
    ]


def toggle_source(notebook_id: str, source_id: str, enabled: bool) -> bool:
    with transaction() as conn:
        cur = conn.execute(
            "UPDATE sources SET enabled = ? WHERE notebook_id = ? AND id = ?",
            (1 if enabled else 0, notebook_id, source_id),
        )
    return cur.rowcount > 0


def get_source_meta(notebook_id: str, source_id: str) -> dict | None:
    r = get_connection().execute(
        "SELECT id, source_type, origin, is_web FROM sources WHERE notebook_id = ? AND id = ?",
        (notebook_id, source_id),
    ).fetchone()
    return dict(r) if r else None


def replace_source_content(
    source_id: str, *, full_text: str, checksum: str,
    chunks: list[tuple[str, int, int]], vectors: np.ndarray,
) -> None:
    """Rebuild a source's chunks in place (used to refresh a live web source)."""
    token_count = sum(tc for _, _, tc in chunks)
    with transaction() as conn:
        conn.execute("DELETE FROM chunks WHERE source_id = ?", (source_id,))
        row = conn.execute("SELECT notebook_id FROM sources WHERE id = ?", (source_id,)).fetchone()
        notebook_id = row["notebook_id"]
        for (text, ordinal, tc), vec in zip(chunks, vectors):
            conn.execute(
                """INSERT INTO chunks (id, source_id, notebook_id, ordinal, text, token_count, vector)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (_new_id("chk"), source_id, notebook_id, ordinal, text, tc,
                 np.asarray(vec, dtype=np.float32).tobytes()),
            )
        conn.execute(
            "UPDATE sources SET full_text = ?, checksum = ?, token_count = ? WHERE id = ?",
            (full_text, checksum, token_count, source_id),
        )


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


_MAX_DRAFT_VERSIONS = 25


def get_or_create_draft(notebook_id: str) -> dict:
    conn = get_connection()
    row = conn.execute("SELECT * FROM drafts WHERE notebook_id = ?", (notebook_id,)).fetchone()
    if row is None:
        draft_id = _new_id("dft")
        created = _now()
        with transaction() as c:
            c.execute(
                "INSERT INTO drafts (id, notebook_id, title, body, updated_at) VALUES (?, ?, ?, ?, ?)",
                (draft_id, notebook_id, "Untitled draft", "", created),
            )
        return {"id": draft_id, "notebook_id": notebook_id, "title": "Untitled draft",
                "body": "", "updated_at": created}
    return dict(row)


def save_draft(
    notebook_id: str, *, title: str | None = None, body: str | None = None,
    snapshot_note: str | None = None,
) -> dict:
    draft = get_or_create_draft(notebook_id)
    with transaction() as conn:
        # Snapshot the current body for undo when an AI edit / restore replaces it.
        if snapshot_note is not None:
            conn.execute(
                "INSERT INTO draft_versions (id, draft_id, body, note, created_at) VALUES (?, ?, ?, ?, ?)",
                (_new_id("ver"), draft["id"], draft["body"], snapshot_note, _now()),
            )
            keep = [r["id"] for r in conn.execute(
                "SELECT id FROM draft_versions WHERE draft_id = ? ORDER BY created_at DESC LIMIT ?",
                (draft["id"], _MAX_DRAFT_VERSIONS),
            ).fetchall()]
            if keep:
                placeholders = ",".join("?" for _ in keep)
                conn.execute(
                    f"DELETE FROM draft_versions WHERE draft_id = ? AND id NOT IN ({placeholders})",
                    (draft["id"], *keep),
                )
        new_title = title if title is not None else draft["title"]
        new_body = body if body is not None else draft["body"]
        conn.execute(
            "UPDATE drafts SET title = ?, body = ?, updated_at = ? WHERE id = ?",
            (new_title, new_body, _now(), draft["id"]),
        )
    return get_or_create_draft(notebook_id)


def list_draft_versions(notebook_id: str) -> list[dict]:
    draft = get_or_create_draft(notebook_id)
    rows = get_connection().execute(
        "SELECT id, note, created_at FROM draft_versions WHERE draft_id = ? ORDER BY created_at DESC",
        (draft["id"],),
    ).fetchall()
    return [dict(r) for r in rows]


def restore_draft_version(notebook_id: str, version_id: str) -> dict | None:
    draft = get_or_create_draft(notebook_id)
    row = get_connection().execute(
        "SELECT body FROM draft_versions WHERE id = ? AND draft_id = ?", (version_id, draft["id"])
    ).fetchone()
    if row is None:
        return None
    return save_draft(notebook_id, body=row["body"], snapshot_note="before restore")


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
        # Default scope: only sources that are switched on.
        rows = conn.execute(
            """SELECT c.id, c.source_id, c.ordinal, c.text, c.vector, s.title AS source_title
               FROM chunks c JOIN sources s ON s.id = c.source_id
               WHERE c.notebook_id = ? AND s.enabled = 1
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
