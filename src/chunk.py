# context prefixes, embeddings, FTS population.
#
# top-level comments get a deterministic prefix. reply comments get a
# one-sentence summary of parent+grandparent via sarvam-30b.
# embeddings are bge-small-en-v1.5 (384-dim), L2-normalized, stored as float32 blobs.

from __future__ import annotations

import asyncio
import traceback
from typing import Iterable

import httpx
import numpy as np

from src import db
from src.config import (
    CONTEXT_PREFIX_MAX_DEPTH,
    MODEL_SMALL,
    P1_CONTEXT_PREFIX,
    PREFIX_LLM_CAP,
    PREFIX_MIN_TEXT_LENGTH_FOR_LLM,
    PREFIX_TIMEOUT_SEC,
)
from src.fetch import log_error
from src.llm import chat, extract_text

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384
PREFIX_CHAR_CAP = 1200  # keep prompts bounded even when parents are long

_embed_model = None


def _embed() :
    # Lazy-load the sentence-transformers model (downloads on first use).
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer

        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model


def _trim(text: str, cap: int = PREFIX_CHAR_CAP) -> str:
    if len(text) <= cap:
        return text
    return text[: cap - 1].rsplit(" ", 1)[0] + "…"


def _load_query_rows(query_id: int) -> tuple[dict[int, dict], dict[int, dict]]:
    # Return (comments_by_id, threads_by_id).
    with db.connect() as conn:
        threads = {
            r["id"]: dict(r)
            for r in conn.execute(
                "SELECT id, title FROM threads WHERE query_id = ?", (query_id,)
            ).fetchall()
        }
        comments = {
            r["id"]: dict(r)
            for r in conn.execute(
                """
                SELECT c.id, c.thread_id, c.parent_id, c.depth, c.text_clean, c.discarded
                FROM comments c JOIN threads t ON t.id = c.thread_id
                WHERE t.query_id = ?
                """,
                (query_id,),
            ).fetchall()
        }
    return comments, threads


def _parent_and_grandparent(
    comment: dict, comments: dict[int, dict], threads: dict[int, dict]
) -> tuple[str, str, bool]:
    # Return (parent_text, grandparent_text, is_top_level).
    # A top-level comment's parent_id points at the story (thread) id; for these
    # we skip the LLM and use a deterministic prefix.
    pid = comment["parent_id"]
    if pid in threads:
        return "", "", True
    parent = comments.get(pid)
    if not parent:
        return "", "", True  # orphan; treat like top-level
    parent_text = _trim(parent.get("text_clean") or "")
    gp = comments.get(parent["parent_id"])
    if gp and gp["id"] not in threads:
        gp_text = _trim(gp.get("text_clean") or "", cap=400)
    else:
        gp_text = ""
    return parent_text, gp_text, False


async def _prefix_for_reply(
    client: httpx.AsyncClient,
    *,
    thread_title: str,
    parent_text: str,
    grandparent_text: str,
    comment_text: str,
) -> str | None:
    prompt = P1_CONTEXT_PREFIX.format(
        thread_title=thread_title,
        grandparent_text=grandparent_text or "",
        parent_text=parent_text or "",
        comment_text=_trim(comment_text, cap=800),
    )
    try:
        resp = await chat(
            [{"role": "user", "content": prompt}],
            purpose="context_prefix",
            model=MODEL_SMALL,
            temperature=0.1,
            # sarvam-30b is a reasoning model so it burns tokens on hidden
            # reasoning before emitting content. 800 was too small; 2500 is safe
            # for a <=25-word sentence plus the reasoning preamble.
            max_tokens=2500,
            client=client,
        )
    except Exception as exc:  # noqa: BLE001
        log_error(
            "chunk._prefix_for_reply",
            f"{type(exc).__name__}: {exc}",
            traceback.format_exc(),
        )
        return None
    text = extract_text(resp)
    if not text:
        return None
    # Keep first non-empty line, drop stray quoting.
    first = text.splitlines()[0].strip().strip('"').strip()
    return first or None


async def generate_prefixes(query_id: int, *, sanity_first: int = 5) -> dict:
    # Generate context prefixes for all non-discarded comments of QUERY_ID.
    # Performance safeguards:
    # - Only the top PREFIX_LLM_CAP comments (ranked by depth, descendants, length)
    # receive LLM-generated prefixes. The rest get fast deterministic ones.
    # - LLM calls are processed in batches of _BATCH_SIZE with progress logging.
    # - The entire step has a hard timeout of PREFIX_TIMEOUT_SEC.
    comments, threads = _load_query_rows(query_id)
    active = [c for c in comments.values() if not c["discarded"]]

    # Partition: trivial (top-level / orphan) vs deep (deterministic) vs LLM-needed
    trivial: list[tuple[int, str]] = []
    to_call: list[dict] = []
    for c in active:
        parent_text, gp_text, top = _parent_and_grandparent(c, comments, threads)
        title = threads[c["thread_id"]]["title"]
        if top:
            trivial.append((c["id"], f"Top-level reply to: {title}"))
        elif c.get("depth", 0) > CONTEXT_PREFIX_MAX_DEPTH:
            # deep comments don't need expensive LLM calls, use deterministic prefix
            trivial.append((c["id"], f"Deep reply in thread: {title}"))
        elif len(c.get("text_clean") or "") >= PREFIX_MIN_TEXT_LENGTH_FOR_LLM:
            # long comments are dense enough to skip LLM prefix generation
            trivial.append((c["id"], f"Reply in thread: {title}"))
        else:
            to_call.append(
                {
                    "id": c["id"],
                    "depth": c.get("depth", 0),
                    "descendant_count": c.get("descendant_count", 0),
                    "text_length": len(c.get("text_clean") or ""),
                    "thread_title": title,
                    "parent_text": parent_text,
                    "grandparent_text": gp_text,
                    "comment_text": c["text_clean"],
                }
            )

    # Prioritize comments that are shallowest, have the most descendants (spark
    # the most discussion), and are longest (most substance).
    if len(to_call) > PREFIX_LLM_CAP:
        to_call.sort(
            key=lambda c: (c["depth"], -c["descendant_count"], -c["text_length"])
        )
        overflow = to_call[PREFIX_LLM_CAP:]
        to_call = to_call[:PREFIX_LLM_CAP]
        # Give the overflow deterministic prefixes
        for c in overflow:
            trivial.append((c["id"], f"Reply in thread: {c['thread_title']}"))
        print(
            f"[chunk] capped LLM prefix calls at {PREFIX_LLM_CAP} "
            f"(skipped {len(overflow)} lower-priority comments)"
        )

    # Write trivial prefixes immediately
    if trivial:
        with db.connect() as conn:
            conn.executemany(
                "UPDATE comments SET context_prefix = ? WHERE id = ?",
                [(p, cid) for cid, p in trivial],
            )

    try:
        await asyncio.wait_for(
            _generate_prefixes_inner(to_call, sanity_first=sanity_first),
            timeout=PREFIX_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        print(
            f"[chunk] prefix generation timed out after {PREFIX_TIMEOUT_SEC}s - "
            "remaining comments get deterministic prefixes"
        )
        # Give any comments that didn't get a prefix a deterministic one
        with db.connect() as conn:
            conn.execute(
                """
                UPDATE comments SET context_prefix = 'Reply in thread (timeout)'
                WHERE context_prefix IS NULL
                  AND discarded = 0
                  AND thread_id IN (SELECT id FROM threads WHERE query_id = ?)
                """,
                (query_id,),
            )

    # Summary
    with db.connect() as conn:
        n_with = conn.execute(
            """
            SELECT COUNT(*) n FROM comments c JOIN threads t ON t.id = c.thread_id
            WHERE t.query_id = ? AND c.discarded = 0 AND c.context_prefix IS NOT NULL
            """,
            (query_id,),
        ).fetchone()["n"]
        n_active = len(active)
    return {"active": n_active, "with_prefix": n_with}


_BATCH_SIZE = 25  # process LLM prefix calls in chunks to avoid overload


async def _generate_prefixes_inner(
    to_call: list[dict], *, sanity_first: int = 5
) -> None:
    # Run LLM prefix generation in batches with progress logging.
    if not to_call:
        return

    async with httpx.AsyncClient() as client:
        # Sanity batch
        if sanity_first and to_call:
            sample = to_call[:sanity_first]
            print(f"[chunk] sanity batch of {len(sample)} prefixes…")
            sample_out = await asyncio.gather(
                *[
                    _prefix_for_reply(
                        client, **{k: v for k, v in s.items() if k not in ("id", "depth", "descendant_count", "text_length")}
                    )
                    for s in sample
                ]
            )
            for s, out in zip(sample, sample_out):
                print(f"  c{s['id']}: {out!r}")
            # Persist sanity
            with db.connect() as conn:
                conn.executemany(
                    "UPDATE comments SET context_prefix = ? WHERE id = ?",
                    [
                        (out or f"Reply in thread: {s['thread_title']}", s["id"])
                        for s, out in zip(sample, sample_out)
                    ],
                )
            remaining = to_call[sanity_first:]
        else:
            remaining = to_call

        if not remaining:
            return

        total = len(remaining)
        print(f"[chunk] generating {total} more prefixes in batches of {_BATCH_SIZE}…")

        for batch_start in range(0, total, _BATCH_SIZE):
            batch = remaining[batch_start : batch_start + _BATCH_SIZE]
            batch_num = batch_start // _BATCH_SIZE + 1
            total_batches = (total + _BATCH_SIZE - 1) // _BATCH_SIZE
            print(f"[chunk]   batch {batch_num}/{total_batches} ({len(batch)} comments)…")

            results = await asyncio.gather(
                *[
                    _prefix_for_reply(
                        client,
                        thread_title=r["thread_title"],
                        parent_text=r["parent_text"],
                        grandparent_text=r["grandparent_text"],
                        comment_text=r["comment_text"],
                    )
                    for r in batch
                ]
            )
            updates: list[tuple[str, int]] = []
            for r, out in zip(batch, results):
                updates.append(
                    (out or f"Reply in thread: {r['thread_title']}", r["id"])
                )
            with db.connect() as conn:
                conn.executemany(
                    "UPDATE comments SET context_prefix = ? WHERE id = ?",
                    updates,
                )


# embeddings


def embed_comments(query_id: int, *, batch_size: int = 32) -> dict:
    # Compute bge-small embeddings over prefix+comment; store as float32 BLOB.
    model = _embed()
    with db.connect() as conn:
        rows = conn.execute(
            """
            SELECT c.id, c.context_prefix, c.text_clean
            FROM comments c JOIN threads t ON t.id = c.thread_id
            WHERE t.query_id = ? AND c.discarded = 0
            ORDER BY c.id
            """,
            (query_id,),
        ).fetchall()

    texts = [
        f"{(r['context_prefix'] or '').strip()}\n\n{r['text_clean']}".strip()
        for r in rows
    ]
    ids = [r["id"] for r in rows]

    if not texts:
        return {"embedded": 0, "dim": EMBED_DIM}

    vecs = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    vecs = np.asarray(vecs, dtype=np.float32)
    assert vecs.shape[1] == EMBED_DIM, f"unexpected dim {vecs.shape}"

    with db.connect() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO embeddings (comment_id, model, vector) VALUES (?,?,?)",
            [(cid, EMBED_MODEL_NAME, vecs[i].tobytes()) for i, cid in enumerate(ids)],
        )
    return {"embedded": len(ids), "dim": EMBED_DIM}


# FTS population


def populate_fts(query_id: int) -> dict:
    # Rebuild the FTS5 index over non-discarded comments.
    # We use the external-content FTS5 `rebuild` command, which reconstructs the
    # index from the backing `comments` table. This is simpler and more robust
    # than per-row DELETE/INSERT (which can trigger "database disk image is
    # malformed" errors if the FTS side is out of sync with the content table).
    with db.connect() as conn:
        # Track just this query's rows for the return count.
        n = conn.execute(
            """
            SELECT COUNT(*) n FROM comments c JOIN threads t ON t.id = c.thread_id
            WHERE t.query_id = ? AND c.discarded = 0
            """,
            (query_id,),
        ).fetchone()["n"]
        conn.execute("INSERT INTO comments_fts(comments_fts) VALUES('rebuild')")
    return {"indexed": n}


# sanity search


def fts_sample(query: str, limit: int = 5) -> list[dict]:
    with db.connect() as conn:
        rows = conn.execute(
            """
            SELECT c.id, c.context_prefix, substr(c.text_clean, 1, 160) snip
            FROM comments_fts JOIN comments c ON c.id = comments_fts.rowid
            WHERE comments_fts MATCH ?
            ORDER BY bm25(comments_fts)
            LIMIT ?
            """,
            (query, limit),
        ).fetchall()
    return [dict(r) for r in rows]
