# claim extraction via sarvam-30b tool calling (batched).
#
# untrusted comment text is wrapped in <hn_comment> tags. system prompt
# tells the model to ignore instructions inside those tags (prompt injection defense).
# pydantic validation on every tool-call output, with split-and-retry on batch failure.

from __future__ import annotations

import asyncio
import json
import math
import traceback
from typing import Literal

import httpx
from pydantic import BaseModel, Field, ValidationError, confloat

from src import db
from src.batching import BatchValidationError, run_with_split_retry
from src.config import (
    EMIT_CLAIMS_BATCH_TOOL,
    EXTRACT_BATCH_SIZE,
    EXTRACT_CANDIDATE_CAP,
    MODEL_SMALL,
    P2_CLAIM_EXTRACTION_BATCH_SYSTEM,
    P2_CLAIM_EXTRACTION_SYSTEM,
)
from src.fetch import log_error
from src.llm import chat, extract_tool_args

# pydantic schemas


class Claim(BaseModel):
    text: str = Field(max_length=400)
    stance: Literal["pro", "con", "neutral", "alternative", "anecdote", "benchmark"]
    category: str
    evidence_type: Literal["anecdote", "benchmark", "citation", "opinion"]
    tools_mentioned: list[str] = []
    confidence: confloat(ge=0, le=1)
    is_firsthand: bool


class ClaimExtraction(BaseModel):
    is_substantive: bool
    claims: list[Claim]


class RouterDecision(BaseModel):
    intent: Literal[
        "pros_cons",
        "performance",
        "comparison",
        "alternatives",
        "how_to",
        "debugging",
        "adoption_risk",
        "consensus_check",
        "follow_up_reference",
        "off_topic",
    ]
    rewritten_query: str
    requires_retrieval: bool
    references_earlier_turn: bool


# Batch pydantic schemas

class BatchClaimResult(BaseModel):
    comment_id: int
    is_substantive: bool
    claims: list[Claim]


class ClaimBatchExtraction(BaseModel):
    results: list[BatchClaimResult]


# tool schema for sarvam tool calling (kept for extract_one wrapper compatibility)

EMIT_CLAIMS_TOOL = {
    "type": "function",
    "function": {
        "name": "emit_claims",
        "description": "Emit structured claims from one HN comment.",
        "parameters": {
            "type": "object",
            "required": ["is_substantive", "claims"],
            "properties": {
                "is_substantive": {"type": "boolean"},
                "claims": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": [
                            "text",
                            "stance",
                            "category",
                            "evidence_type",
                            "tools_mentioned",
                            "confidence",
                            "is_firsthand",
                        ],
                        "properties": {
                            "text": {"type": "string"},
                            "stance": {
                                "type": "string",
                                "enum": [
                                    "pro",
                                    "con",
                                    "neutral",
                                    "alternative",
                                    "anecdote",
                                    "benchmark",
                                ],
                            },
                            "category": {"type": "string"},
                            "evidence_type": {
                                "type": "string",
                                "enum": ["anecdote", "benchmark", "citation", "opinion"],
                            },
                            "tools_mentioned": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "confidence": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                            },
                            "is_firsthand": {"type": "boolean"},
                        },
                    },
                },
            },
        },
    },
}


# candidate selection - pick the highest-signal comments for extraction


def _signal_score(row) -> float:
    d = row["depth"]
    text_len = row["text_length"]
    desc = row["descendant_count"]
    has_code = row["has_code"]
    score = (
        0.5 * math.log1p(text_len / 200.0)
        + 0.7 * math.log1p(desc)
        + 0.3 * math.log1p(d)
        + (0.5 if has_code else 0.0)
    )
    if d > 5:
        score -= 0.2 * (d - 5)
    return score


def select_candidates(query_id: int, cap: int = EXTRACT_CANDIDATE_CAP) -> list[int]:
    with db.connect() as conn:
        rows = conn.execute(
            """
            SELECT c.id, c.depth, c.text_length, c.descendant_count, c.has_code
            FROM comments c JOIN threads t ON t.id = c.thread_id
            WHERE t.query_id = ? AND c.discarded = 0
            """,
            (query_id,),
        ).fetchall()
    if len(rows) <= cap:
        return [r["id"] for r in rows]
    scored = sorted(rows, key=_signal_score, reverse=True)[:cap]
    return [r["id"] for r in scored]


# --- Batch user message builder ---


def _build_user_message(comment_id: int, context_prefix: str, text_clean: str) -> str:
    # Per P2: wrap in <hn_comment id="..."> tags; the context prefix is ours
    # (trusted), so put it before the tags. Comment text is untrusted.
    prefix = (context_prefix or "").replace("\n", " ").strip()
    return (
        f"Context: {prefix}\n\n"
        f'<hn_comment id="{comment_id}">\n{text_clean}\n</hn_comment>'
    )


def _build_extract_batch_user_msg(items: list[dict]) -> str:
    """items: each dict has keys comment_id, context_prefix, text_clean."""
    parts: list[str] = []
    for it in items:
        prefix = (it.get("context_prefix") or "").replace("\n", " ").strip()
        parts.append(
            f"Context: {prefix}\n"
            f'<hn_comment id="{it["comment_id"]}">\n'
            f'{it["text_clean"]}\n'
            f'</hn_comment>'
        )
    return "\n\n".join(parts)


def _validate(args: dict | None) -> ClaimExtraction | None:
    if args is None:
        return None
    try:
        return ClaimExtraction.model_validate(args)
    except ValidationError:
        return None


# --- Batch runner ---


async def _run_extract_batch(
    items: list[dict], client: httpx.AsyncClient, temp: float = 0.1
) -> dict[int, BatchClaimResult]:
    """Returns {comment_id: BatchClaimResult}.
    Raises BatchValidationError on schema/id-set/truncation issues."""
    if not items:
        return {}

    items = sorted(items, key=lambda x: x["comment_id"])
    input_ids = {it["comment_id"] for it in items}

    user_msg = _build_extract_batch_user_msg(items)
    messages = [
        {"role": "system", "content": P2_CLAIM_EXTRACTION_BATCH_SYSTEM},
        {"role": "user", "content": user_msg},
    ]

    resp = await chat(
        messages,
        purpose="claim_extract_batch",
        model=MODEL_SMALL,
        tools=[EMIT_CLAIMS_BATCH_TOOL],
        tool_choice={"type": "function", "function": {"name": "emit_claims_batch"}},
        temperature=temp,
        max_tokens=4000,
        client=client,
    )

    finish = (resp.get("choices") or [{}])[0].get("finish_reason")
    if finish == "length":
        raise BatchValidationError(f"extract batch truncated (n={len(items)})")

    args = extract_tool_args(resp, "emit_claims_batch")
    if args is None:
        raise BatchValidationError("emit_claims_batch tool call missing")

    try:
        parsed = ClaimBatchExtraction.model_validate(args)
    except ValidationError as e:
        raise BatchValidationError(f"extract schema invalid: {e}") from e

    out: dict[int, BatchClaimResult] = {}
    for r in parsed.results:
        if r.comment_id not in input_ids:
            log_error(
                "extract._run_extract_batch",
                f"hallucinated comment_id {r.comment_id}",
                "",
                inp=f"input_size={len(input_ids)}",
            )
            continue
        if r.comment_id in out:
            continue  # first wins
        out[r.comment_id] = r

    missing = input_ids - out.keys()
    if missing:
        raise BatchValidationError(
            f"extract batch incomplete: missing {len(missing)} of {len(input_ids)}"
        )
    return out


# --- extract_one: compatibility wrapper for tests ---


async def extract_one(
    client: httpx.AsyncClient,
    *,
    comment_id: int,
    context_prefix: str,
    text_clean: str,
) -> ClaimExtraction | None:
    """Compatibility wrapper around _run_extract_batch for single-comment use.
    Used by tests/test_extract.py and any external single-comment callers."""
    item = {
        "comment_id": comment_id,
        "context_prefix": context_prefix,
        "text_clean": text_clean,
    }
    try:
        out = await _run_extract_batch([item], client, temp=0.1)
    except BatchValidationError:
        try:
            out = await _run_extract_batch([item], client, temp=0.0)
        except BatchValidationError:
            log_error(
                "extract.extract_one",
                "validation failed twice in batch wrapper; skipping",
                "",
                inp=f"comment_id={comment_id}",
            )
            return None
    r = out.get(comment_id)
    if r is None:
        return None
    return ClaimExtraction(is_substantive=r.is_substantive, claims=r.claims)


# --- Batched extract_all ---


async def extract_all(query_id: int, *, sanity_first: int = 5) -> dict:
    cand_ids = select_candidates(query_id)

    with db.connect() as conn:
        rows = {
            r["comment_id"]: dict(r)
            for r in conn.execute(
                """
                SELECT c.id AS comment_id, c.context_prefix, c.text_clean
                FROM comments c WHERE c.id IN ({})
                """.format(",".join("?" * len(cand_ids))),
                cand_ids,
            ).fetchall()
        }
        already = {
            r["comment_id"]
            for r in conn.execute(
                "SELECT DISTINCT comment_id FROM claims WHERE comment_id IN ({})".format(
                    ",".join("?" * len(cand_ids))
                ),
                cand_ids,
            ).fetchall()
        }

    todo_items = [rows[cid] for cid in cand_ids if cid not in already and cid in rows]

    async with httpx.AsyncClient() as client:
        # Sanity batch
        if sanity_first and todo_items:
            sample = todo_items[:sanity_first]
            print(f"[extract] sanity batch of {len(sample)}…")
            sample_out = await run_with_split_retry(
                sample,
                item_id=lambda r: r["comment_id"],
                run_batch=lambda b: _run_extract_batch(b, client, temp=0.1),
                on_single_failure=lambda r: None,  # drop -> caller marks non-substantive
                label="extract.sanity",
            )
            _print_extract_sample(sample, sample_out)
            _persist_extract_batch(sample, sample_out)
            todo_items = todo_items[sanity_first:]

        # Main run
        if todo_items:
            n_batches = (len(todo_items) + EXTRACT_BATCH_SIZE - 1) // EXTRACT_BATCH_SIZE
            print(f"[extract] extracting {len(todo_items)} comments in {n_batches} batches…")
            batches = [
                todo_items[i : i + EXTRACT_BATCH_SIZE]
                for i in range(0, len(todo_items), EXTRACT_BATCH_SIZE)
            ]
            outs = await asyncio.gather(*[
                run_with_split_retry(
                    b,
                    item_id=lambda r: r["comment_id"],
                    run_batch=lambda batch, c=client: _run_extract_batch(batch, c, temp=0.1),
                    on_single_failure=lambda r: None,
                    label=f"extract.b{idx}",
                )
                for idx, b in enumerate(batches)
            ])
            for batch, out in zip(batches, outs):
                _persist_extract_batch(batch, out)

    # Summary unchanged
    with db.connect() as conn:
        n_claims = conn.execute("SELECT COUNT(*) n FROM claims").fetchone()["n"]
        n_sub = conn.execute(
            """
            SELECT COUNT(*) n FROM comments c JOIN threads t ON t.id=c.thread_id
            WHERE t.query_id=? AND c.is_substantive=1
            """,
            (query_id,),
        ).fetchone()["n"]
    return {"scanned": len(cand_ids), "substantive": n_sub, "claims": n_claims}


def _print_extract_sample(sample: list[dict], out: dict[int, BatchClaimResult]) -> None:
    for s in sample:
        cid = s["comment_id"]
        r = out.get(cid)
        if r is None:
            print(f"  c{cid}: FAILED")
        else:
            print(
                f"  c{cid}: substantive={r.is_substantive} n_claims={len(r.claims)} "
                f"stances={[c.stance for c in r.claims]}"
            )


def _persist_extract_batch(
    batch: list[dict], out: dict[int, BatchClaimResult]
) -> None:
    """Write batch results to DB. Items missing from `out` are marked non-substantive
    (matches the previous double-failure-skip behavior)."""
    with db.connect() as conn:
        for item in batch:
            cid = item["comment_id"]
            r = out.get(cid)
            if r is None:
                conn.execute(
                    "UPDATE comments SET is_substantive = 0 WHERE id = ?", (cid,)
                )
                continue
            conn.execute(
                "UPDATE comments SET is_substantive = ? WHERE id = ?",
                (1 if r.is_substantive else 0, cid),
            )
            if not r.is_substantive:
                continue
            conn.executemany(
                """
                INSERT INTO claims (comment_id, claim_text, stance, category,
                                    evidence_type, tools_mentioned, confidence,
                                    is_firsthand)
                VALUES (?,?,?,?,?,?,?,?)
                """,
                [
                    (
                        cid,
                        cl.text,
                        cl.stance,
                        cl.category,
                        cl.evidence_type,
                        json.dumps(cl.tools_mentioned, ensure_ascii=False),
                        float(cl.confidence),
                        1 if cl.is_firsthand else 0,
                    )
                    for cl in r.claims
                ],
            )
