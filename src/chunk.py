# context prefixes, embeddings, FTS population.
#
# top-level comments get a deterministic prefix. reply comments get a
# one-sentence summary of parent+grandparent via sarvam-30b (batched).
# embeddings are bge-small-en-v1.5 (384-dim), L2-normalized, stored as float32 blobs.

from __future__ import annotations

import asyncio
import traceback
from typing import Iterable

import httpx
import numpy as np
from pydantic import BaseModel, ValidationError

from src import db
from src.batching import BatchValidationError, run_with_split_retry
from src.config import (
    CONTEXT_PREFIX_MAX_DEPTH,
    EMIT_PREFIXES_BATCH_TOOL,
    MODEL_SMALL,
    P1_CONTEXT_PREFIX,
    P1_CONTEXT_PREFIX_BATCH,
    PREFIX_BATCH_SIZE,
    PREFIX_LLM_CAP,
    PREFIX_MIN_TEXT_LENGTH_FOR_LLM,
    PREFIX_TIMEOUT_SEC,
)
from src.fetch import log_error
from src.llm import chat, extract_tool_args

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


# --- Pydantic schemas for batched prefix validation ---

class _PrefixItem(BaseModel):
    comment_id: int
    prefix: str

class _PrefixBatch(BaseModel):
    prefixes: list[_PrefixItem]


# --- Batched prefix generation ---


def _build_prefix_batch_user_msg(items: list[dict]) -> str:
    """items: each dict has keys id, thread_title, parent_text,
    grandparent_text, comment_text."""
    parts: list[str] = []
    for it in items:
        parent = (it.get("parent_text") or "").strip()
        gp = (it.get("grandparent_text") or "").strip()
        body = _trim(it["comment_text"], cap=800)
        title = it["thread_title"]
        parts.append(
            f'<hn_comment id="{it["id"]}">\n'
            f'Thread title: {title}\n'
            f'Grandparent (may be empty): <gp>{gp}</gp>\n'
            f'Parent: <p>{parent}</p>\n'
            f'This reply: <r>{body}</r>\n'
            f'</hn_comment>'
        )
    return "\n\n".join(parts)


async def _run_prefix_batch(
    items: list[dict], client: httpx.AsyncClient
) -> dict[int, str]:
    """Call the LLM for one batch of comments. Returns {comment_id: prefix}.
    Raises BatchValidationError on any structural / id-set / truncation issue.
    """
    if not items:
        return {}

    # Sort by id for stable cache keys (see §3.8).
    items = sorted(items, key=lambda x: x["id"])
    input_ids = {it["id"] for it in items}

    msg_user = _build_prefix_batch_user_msg(items)
    messages = [
        {"role": "system", "content": P1_CONTEXT_PREFIX_BATCH},
        {"role": "user", "content": msg_user},
    ]

    resp = await chat(
        messages,
        purpose="context_prefix_batch",
        model=MODEL_SMALL,
        tools=[EMIT_PREFIXES_BATCH_TOOL],
        tool_choice={"type": "function", "function": {"name": "emit_prefixes"}},
        temperature=0.1,
        max_tokens=4000,
        client=client,
    )

    # Truncation -> definitely incomplete.
    finish = (resp.get("choices") or [{}])[0].get("finish_reason")
    if finish == "length":
        raise BatchValidationError(f"prefix batch truncated (finish=length, n={len(items)})")

    args = extract_tool_args(resp, "emit_prefixes")
    if args is None:
        raise BatchValidationError("emit_prefixes tool call missing")

    try:
        parsed = _PrefixBatch.model_validate(args)
    except ValidationError as e:
        raise BatchValidationError(f"prefix schema invalid: {e}") from e

    out: dict[int, str] = {}
    for entry in parsed.prefixes:
        if entry.comment_id not in input_ids:
            # Hallucinated id — log and drop, but don't fail the batch yet
            # (other ids may still be valid).
            log_error(
                "chunk._run_prefix_batch",
                f"hallucinated comment_id {entry.comment_id} not in input set",
                "",
                inp=f"input_ids_size={len(input_ids)}",
            )
            continue
        # First win on duplicates.
        if entry.comment_id in out:
            continue
        prefix = entry.prefix.strip().strip('"').strip()
        if prefix:
            out[entry.comment_id] = prefix

    # Now check the bijection. Missing ids => incomplete batch => split.
    missing = input_ids - out.keys()
    if missing:
        raise BatchValidationError(
            f"prefix batch incomplete: missing {len(missing)} of {len(input_ids)} ids"
        )
    return out


async def generate_prefixes(query_id: int, *, sanity_first: int = 5) -> dict:
    """Generate context prefixes for all non-discarded comments of QUERY_ID
    that don't already have one (idempotent across runs)."""
    comments, threads = _load_query_rows(query_id)

    # Skip comments that already have a prefix from a prior run.
    with db.connect() as conn:
        rows = conn.execute(
            """
            SELECT c.id FROM comments c JOIN threads t ON t.id = c.thread_id
            WHERE t.query_id = ? AND c.context_prefix IS NOT NULL AND c.discarded = 0
            """,
            (query_id,),
        ).fetchall()
        already = {r["id"] for r in rows}

    active = [c for c in comments.values() if not c["discarded"] and c["id"] not in already]

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
        n_active = len(active) + len(already)
    return {"active": n_active, "with_prefix": n_with}


async def _generate_prefixes_inner(
    to_call: list[dict], *, sanity_first: int = 5
) -> None:
    """Run batched LLM prefix generation with progress logging."""
    if not to_call:
        return

    async with httpx.AsyncClient() as client:
        # Sanity batch (first N as one batch, printed for human review)
        if sanity_first and to_call:
            sample = to_call[:sanity_first]
            print(f"[chunk] sanity batch of {len(sample)} prefixes…")
            sample_out = await run_with_split_retry(
                sample,
                item_id=lambda c: c["id"],
                run_batch=lambda batch: _run_prefix_batch(batch, client),
                on_single_failure=lambda c: f"Reply in thread: {c['thread_title']}",
                label="prefix.sanity",
            )
            for s in sample:
                print(f"  c{s['id']}: {sample_out.get(s['id'], '(fallback)')!r}")
            _persist_prefixes(sample_out, sample)
            remaining = to_call[sanity_first:]
        else:
            remaining = to_call

        if not remaining:
            return

        total = len(remaining)
        n_batches = (total + PREFIX_BATCH_SIZE - 1) // PREFIX_BATCH_SIZE
        print(f"[chunk] generating {total} prefixes across {n_batches} batches of "
              f"up to {PREFIX_BATCH_SIZE}…")

        # Submit all batches concurrently; LLM_CONCURRENCY semaphore in llm.py
        # bounds parallelism.
        batches = [
            remaining[i : i + PREFIX_BATCH_SIZE]
            for i in range(0, total, PREFIX_BATCH_SIZE)
        ]
        results_per_batch = await asyncio.gather(*[
            run_with_split_retry(
                b,
                item_id=lambda c: c["id"],
                run_batch=lambda batch, _c=client: _run_prefix_batch(batch, _c),
                on_single_failure=lambda c: f"Reply in thread: {c['thread_title']}",
                label=f"prefix.b{idx}",
            )
            for idx, b in enumerate(batches)
        ])

        for batch, result_dict in zip(batches, results_per_batch):
            _persist_prefixes(result_dict, batch)


def _persist_prefixes(result_dict: dict[int, str], batch: list[dict]) -> None:
    """Write prefixes to DB. Items without a result get the deterministic fallback."""
    updates = []
    for it in batch:
        prefix = result_dict.get(it["id"]) or f"Reply in thread: {it['thread_title']}"
        updates.append((prefix, it["id"]))
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
