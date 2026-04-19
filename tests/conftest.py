# shared test fixtures

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from unittest.mock import patch

import pytest

from src import db


@pytest.fixture()
def tmp_db(tmp_path: Path):
    # Create a fresh in-memory-style SQLite DB with the full schema.
    # Patches ``db.connect`` and ``db.DB_PATH`` so all production code
    # transparently reads / writes this temporary database.
    db_path = tmp_path / "test_hn.db"
    db.init_db(db_path)

    @contextmanager
    def _connect(path: Path = db_path) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    with patch.object(db, "connect", _connect), patch.object(db, "DB_PATH", db_path):
        yield db_path


def seed_query(db_path: Path, topic: str = "test topic") -> int:
    # Insert a minimal query row and return its id.
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.execute(
        "INSERT INTO queries (topic, fetched_at, config_json) VALUES (?,?,?)",
        (topic, 1000000, "{}"),
    )
    qid = cur.lastrowid
    conn.commit()
    conn.close()
    return qid


def seed_thread(db_path: Path, query_id: int, thread_id: int, title: str = "Test Thread", slot: str = "relevance", points: int = 100) -> None:
    # Insert a minimal thread row.
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """INSERT OR REPLACE INTO threads
           (id, query_id, title, url, points, num_comments, created_at, author, slot, raw_json)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (thread_id, query_id, title, "https://example.com", points, 50, 1000000, "testuser", slot, "{}"),
    )
    conn.commit()
    conn.close()


def seed_comment(
    db_path: Path,
    comment_id: int,
    thread_id: int,
    *,
    parent_id: int | None = None,
    text_clean: str = "This is a test comment with enough length to pass filters",
    depth: int = 0,
    discarded: int = 0,
    has_code: int = 0,
    text_length: int | None = None,
    is_substantive: int | None = None,
    context_prefix: str | None = None,
) -> None:
    # Insert a comment row.
    if text_length is None:
        text_length = len(text_clean)
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """INSERT OR REPLACE INTO comments
           (id, thread_id, parent_id, author, created_at, text, text_clean,
            depth, descendant_count, text_length, has_code, quote_density,
            context_prefix, is_substantive, discarded, discard_reason)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (comment_id, thread_id, parent_id or thread_id, "testauthor", 1000000,
         text_clean, text_clean, depth, 0, text_length, has_code, 0.0,
         context_prefix, is_substantive, discarded, None),
    )
    conn.commit()
    conn.close()


def seed_claim(
    db_path: Path,
    comment_id: int,
    *,
    claim_text: str = "Test claim text",
    stance: str = "pro",
    category: str = "general",
    evidence_type: str = "opinion",
    tools_mentioned: str = "[]",
    confidence: float = 0.8,
    is_firsthand: int = 0,
) -> int:
    # Insert a claim row and return its id.
    conn = sqlite3.connect(str(db_path))
    cur = conn.execute(
        """INSERT INTO claims
           (comment_id, claim_text, stance, category, evidence_type,
            tools_mentioned, confidence, is_firsthand)
           VALUES (?,?,?,?,?,?,?,?)""",
        (comment_id, claim_text, stance, category, evidence_type,
         tools_mentioned, confidence, is_firsthand),
    )
    claim_id = cur.lastrowid
    conn.commit()
    conn.close()
    return claim_id


def seed_cluster(
    db_path: Path,
    query_id: int,
    *,
    label: str = "Test cluster",
    stance: str = "pro",
    category: str = "general",
    weight: float = 5.0,
    member_claim_ids: list[int] | None = None,
) -> int:
    # Insert a cluster row and return its id.
    import json
    conn = sqlite3.connect(str(db_path))
    cur = conn.execute(
        """INSERT INTO clusters
           (query_id, label, stance, category, weight, member_claim_ids)
           VALUES (?,?,?,?,?,?)""",
        (query_id, label, stance, category, weight,
         json.dumps(member_claim_ids or [])),
    )
    cid = cur.lastrowid
    conn.commit()
    conn.close()
    return cid
