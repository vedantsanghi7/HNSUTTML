# hybrid retrieval: BM25 + dense -> RRF fusion -> feature re-score.
#
# BM25 via sqlite FTS5 (top 30), dense via cosine sim on bge-small (top 30),
# fused with reciprocal rank fusion (k=60) -> top 40, then re-scored with
# community signals (thread points, descendants, firsthand experience) -> top 8.

from __future__ import annotations

import logging
import math
import re
from typing import Iterable

import numpy as np

from src import db
from src.chunk import _embed

logger = logging.getLogger(__name__)

RRF_K = 60
BM25_LIMIT = 30
DENSE_LIMIT = 30
FUSED_LIMIT = 40
FINAL_LIMIT = 8


# FTS query sanitization

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+\-]*")


def _fts_query(raw: str) -> str:
    # Turn a free-text query into an FTS5-safe OR query.
    # FTS5 MATCH syntax is picky about special characters. We lowercase, extract
    # alphanumeric tokens, stem away trivial stopwords, and produce a quoted OR
    # expression so that operators like `:` or `(` in user input never leak into
    # the FTS parser.
    tokens = [t.lower() for t in _TOKEN_RE.findall(raw)]
    stop = {
        "the", "and", "for", "with", "that", "this", "from", "into", "its",
        "are", "was", "been", "has", "have", "not", "but", "about", "use",
        "how", "why", "what", "can", "should", "does", "will", "did", "did",
        "a", "an", "in", "on", "to", "of", "is", "it", "be", "or", "as", "if",
        "say", "says", "said", "do", "does", "get", "gets", "go", "goes",
    }
    tokens = [t for t in tokens if t not in stop and len(t) >= 2]
    if not tokens:
        return '""'  # matches nothing; caller falls back to dense
    # Quote each token to escape any FTS syntax inside it; OR them together.
    return " OR ".join(f'"{t}"' for t in tokens)


# BM25


def bm25_search(query: str, query_id: int, limit: int = BM25_LIMIT) -> list[tuple[int, float]]:
    # Return [(comment_id, bm25_score)]. Lower BM25 = better match.
    fts_q = _fts_query(query)
    if fts_q == '""':
        return []
    sql = """
        SELECT c.id AS cid, bm25(comments_fts) AS score
        FROM comments_fts
        JOIN comments c ON c.id = comments_fts.rowid
        JOIN threads t ON t.id = c.thread_id
        WHERE comments_fts MATCH ?
          AND t.query_id = ?
          AND c.discarded = 0
        ORDER BY score
        LIMIT ?
    """
    try:
        with db.connect() as conn:
            rows = conn.execute(sql, (fts_q, query_id, limit)).fetchall()
    except Exception as e:  # noqa: BLE001
        logger.warning("bm25 fallback (empty) due to FTS error: %s", e)
        return []
    return [(int(r["cid"]), float(r["score"])) for r in rows]


# dense retrieval


def _load_embeddings(query_id: int) -> tuple[list[int], np.ndarray]:
    # Load all (comment_id, vector) pairs for a query. Vectors are L2-normalized.
    with db.connect() as conn:
        rows = conn.execute(
            """
            SELECT e.comment_id AS cid, e.vector AS vec
            FROM embeddings e
            JOIN comments c ON c.id = e.comment_id
            JOIN threads t ON t.id = c.thread_id
            WHERE t.query_id = ? AND c.discarded = 0
            """,
            (query_id,),
        ).fetchall()
    if not rows:
        return [], np.zeros((0, 384), dtype=np.float32)
    ids = [int(r["cid"]) for r in rows]
    vecs = np.stack(
        [np.frombuffer(r["vec"], dtype=np.float32) for r in rows], axis=0
    )
    return ids, vecs


def dense_search(query: str, query_id: int, limit: int = DENSE_LIMIT) -> list[tuple[int, float]]:
    # Return [(comment_id, cosine_sim)]. Higher = better.
    ids, vecs = _load_embeddings(query_id)
    if not ids:
        return []
    model = _embed()
    qv = model.encode([query], normalize_embeddings=True)[0].astype(np.float32)
    # vectors are L2-normalized, so dot product == cosine similarity.
    sims = vecs @ qv
    top = np.argsort(-sims)[:limit]
    return [(ids[i], float(sims[i])) for i in top]


# fusion


def rrf_fuse(
    bm25_hits: list[tuple[int, float]],
    dense_hits: list[tuple[int, float]],
    k: int = RRF_K,
    limit: int = FUSED_LIMIT,
) -> list[tuple[int, float]]:
    # Reciprocal Rank Fusion. Returns [(comment_id, rrf_score)] top-limit.
    scores: dict[int, float] = {}
    for rank, (cid, _) in enumerate(bm25_hits):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    for rank, (cid, _) in enumerate(dense_hits):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:limit]
    return fused


# re-score


def _fetch_rescore_features(
    cids: list[int], query_id: int
) -> dict[int, dict]:
    # Per-comment features for the re-scorer.
    if not cids:
        return {}
    placeholders = ",".join("?" * len(cids))
    sql = f"""
        SELECT c.id AS cid,
               c.descendant_count AS desc_n,
               c.text_clean AS text_clean,
               c.author AS author,
               c.thread_id AS thread_id,
               c.depth AS depth,
               t.points AS thread_points,
               t.title AS thread_title,
               EXISTS(SELECT 1 FROM claims cl
                      WHERE cl.comment_id = c.id AND cl.is_firsthand = 1) AS has_firsthand
        FROM comments c
        JOIN threads t ON t.id = c.thread_id
        WHERE c.id IN ({placeholders})
          AND t.query_id = ?
          AND c.discarded = 0
    """
    with db.connect() as conn:
        rows = conn.execute(sql, cids + [query_id]).fetchall()
    return {int(r["cid"]): dict(r) for r in rows}


def rescore(
    fused: list[tuple[int, float]], query_id: int, top_n: int = FINAL_LIMIT
) -> list[dict]:
    # Apply the 3-feature re-score; return top_n rich rows ready for prompting.
    if not fused:
        return []
    cids = [c for c, _ in fused]
    feats = _fetch_rescore_features(cids, query_id)
    scored: list[dict] = []
    for cid, rrf_score in fused:
        f = feats.get(cid)
        if f is None:
            continue
        final = (
            rrf_score
            + 0.3 * math.log1p(f["thread_points"] or 0)
            + 0.2 * math.log1p(f["desc_n"] or 0)
            + 0.4 * (1.0 if f["has_firsthand"] else 0.0)
        )
        scored.append({**f, "rrf": rrf_score, "final": final, "cid": cid})
    scored.sort(key=lambda r: r["final"], reverse=True)
    return scored[:top_n]


# public entry point


def hybrid_retrieve(query: str, query_id: int, top_n: int = FINAL_LIMIT) -> list[dict]:
    # End-to-end hybrid retrieval for a rewritten user query.
    # Returns a list of dicts with keys: cid, text_clean, author, thread_title,
    # thread_points, desc_n, has_firsthand, rrf, final.
    bm = bm25_search(query, query_id)
    dn = dense_search(query, query_id)
    if not bm and not dn:
        return []
    fused = rrf_fuse(bm, dn)
    return rescore(fused, query_id, top_n=top_n)
