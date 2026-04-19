# sqlite schema + connection helpers.
# single db at data/hn.db. FTS5 for BM25, embeddings stored as float32 blobs.

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from src.config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS queries (
  id INTEGER PRIMARY KEY,
  topic TEXT NOT NULL,
  fetched_at INTEGER NOT NULL,
  config_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS threads (
  id INTEGER PRIMARY KEY,
  query_id INTEGER NOT NULL REFERENCES queries(id),
  title TEXT NOT NULL,
  url TEXT,
  points INTEGER,
  num_comments INTEGER,
  created_at INTEGER NOT NULL,
  author TEXT,
  slot TEXT NOT NULL,
  raw_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS comments (
  id INTEGER PRIMARY KEY,
  thread_id INTEGER NOT NULL REFERENCES threads(id),
  parent_id INTEGER,
  author TEXT,
  created_at INTEGER,
  text TEXT NOT NULL,
  text_clean TEXT NOT NULL,
  depth INTEGER NOT NULL,
  descendant_count INTEGER NOT NULL,
  text_length INTEGER NOT NULL,
  has_code INTEGER NOT NULL,
  quote_density REAL NOT NULL,
  context_prefix TEXT,
  is_substantive INTEGER,
  discarded INTEGER DEFAULT 0,
  discard_reason TEXT
);
CREATE INDEX IF NOT EXISTS idx_comments_thread ON comments(thread_id);
CREATE INDEX IF NOT EXISTS idx_comments_parent ON comments(parent_id);

CREATE VIRTUAL TABLE IF NOT EXISTS comments_fts USING fts5(
  text_clean, context_prefix,
  content='comments', content_rowid='id'
);

CREATE TABLE IF NOT EXISTS embeddings (
  comment_id INTEGER PRIMARY KEY REFERENCES comments(id),
  model TEXT NOT NULL,
  vector BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS claims (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  comment_id INTEGER NOT NULL REFERENCES comments(id),
  claim_text TEXT NOT NULL,
  stance TEXT NOT NULL,
  category TEXT NOT NULL,
  evidence_type TEXT NOT NULL,
  tools_mentioned TEXT,
  confidence REAL NOT NULL,
  is_firsthand INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_claims_comment ON claims(comment_id);

CREATE TABLE IF NOT EXISTS claim_embeddings (
  claim_id INTEGER PRIMARY KEY REFERENCES claims(id),
  vector BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS clusters (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  query_id INTEGER NOT NULL REFERENCES queries(id),
  label TEXT NOT NULL,
  stance TEXT,
  category TEXT,
  weight REAL NOT NULL,
  member_claim_ids TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS llm_calls (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts INTEGER NOT NULL,
  model TEXT NOT NULL,
  purpose TEXT NOT NULL,
  tokens_in INTEGER,
  tokens_out INTEGER,
  latency_ms INTEGER,
  cache_hit INTEGER NOT NULL,
  prompt_sha TEXT NOT NULL
);
"""


def _connect(path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_db(path: Path = DB_PATH) -> None:
    # Create schema if absent. Idempotent.
    path.parent.mkdir(parents=True, exist_ok=True)
    with _connect(path) as conn:
        conn.executescript(SCHEMA)
        conn.commit()


@contextmanager
def connect(path: Path = DB_PATH) -> Iterator[sqlite3.Connection]:
    # Yield a connection with foreign_keys on and auto-commit on exit.
    conn = _connect(path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()
