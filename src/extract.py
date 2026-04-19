# claim extraction via sarvam-30b tool calling.
#
# untrusted comment text is wrapped in <hn_comment> tags. system prompt
# tells the model to ignore instructions inside those tags (prompt injection defense).
# pydantic validation on every tool-call output, retry once at temp=0 on failure.

from __future__ import annotations

import asyncio
import json
import math
import traceback
from typing import Literal

import httpx
from pydantic import BaseModel, Field, ValidationError, confloat

from src import db
from src.config import EXTRACT_CANDIDATE_CAP, MODEL_SMALL, P2_CLAIM_EXTRACTION_SYSTEM
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


# tool schema for sarvam tool calling

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


# extraction


def _build_user_message(comment_id: int, context_prefix: str, text_clean: str) -> str:
    # Per P2: wrap in <hn_comment id="..."> tags; the context prefix is ours
    # (trusted), so put it before the tags. Comment text is untrusted.
    prefix = (context_prefix or "").replace("\n", " ").strip()
    return (
        f"Context: {prefix}\n\n"
        f'<hn_comment id="{comment_id}">\n{text_clean}\n</hn_comment>'
    )


def _validate(args: dict | None) -> ClaimExtraction | None:
    if args is None:
        return None
    try:
        return ClaimExtraction.model_validate(args)
    except ValidationError:
        return None


async def extract_one(
    client: httpx.AsyncClient,
    *,
    comment_id: int,
    context_prefix: str,
    text_clean: str,
) -> ClaimExtraction | None:
    user_msg = _build_user_message(comment_id, context_prefix, text_clean)
    messages = [
        {"role": "system", "content": P2_CLAIM_EXTRACTION_SYSTEM},
        {"role": "user", "content": user_msg},
    ]

    for attempt, temp in enumerate((0.1, 0.0)):
        try:
            resp = await chat(
                messages,
                purpose="claim_extract",
                model=MODEL_SMALL,
                tools=[EMIT_CLAIMS_TOOL],
                tool_choice={"type": "function", "function": {"name": "emit_claims"}},
                temperature=temp,
                max_tokens=3500,
                client=client,
            )
        except Exception as exc:  # noqa: BLE001
            log_error(
                "extract.extract_one",
                f"attempt {attempt}: {type(exc).__name__}: {exc}",
                traceback.format_exc(),
                inp=f"comment_id={comment_id}",
            )
            continue
        args = extract_tool_args(resp, "emit_claims")
        parsed = _validate(args)
        if parsed is not None:
            return parsed
    log_error(
        "extract.extract_one",
        "validation failed twice; skipping",
        "",
        inp=f"comment_id={comment_id}",
    )
    return None


async def extract_all(query_id: int, *, sanity_first: int = 5) -> dict:
    cand_ids = select_candidates(query_id)

    with db.connect() as conn:
        rows = {
            r["id"]: dict(r)
            for r in conn.execute(
                """
                SELECT c.id, c.context_prefix, c.text_clean
                FROM comments c WHERE c.id IN ({})
                """.format(",".join("?" * len(cand_ids))),
                cand_ids,
            ).fetchall()
        }
        # Skip comments we've already extracted (idempotency across runs).
        already = {
            r["comment_id"]
            for r in conn.execute(
                "SELECT DISTINCT comment_id FROM claims WHERE comment_id IN ({})".format(
                    ",".join("?" * len(cand_ids))
                ),
                cand_ids,
            ).fetchall()
        }

    todo = [cid for cid in cand_ids if cid not in already]

    async with httpx.AsyncClient() as client:
        # Sanity first
        if sanity_first and todo:
            sample_ids = todo[:sanity_first]
            print(f"[extract] sanity batch of {len(sample_ids)}…")
            sample = await asyncio.gather(
                *[
                    extract_one(
                        client,
                        comment_id=cid,
                        context_prefix=rows[cid]["context_prefix"] or "",
                        text_clean=rows[cid]["text_clean"],
                    )
                    for cid in sample_ids
                ]
            )
            for cid, out in zip(sample_ids, sample):
                if out is None:
                    print(f"  c{cid}: FAILED")
                else:
                    print(
                        f"  c{cid}: substantive={out.is_substantive} n_claims={len(out.claims)} "
                        f"stances={[c.stance for c in out.claims]}"
                    )
            # Persist sanity batch
            _persist_batch(query_id, list(zip(sample_ids, sample)))
            todo = todo[sanity_first:]

        if todo:
            print(f"[extract] extracting {len(todo)} more comments…")
            results = await asyncio.gather(
                *[
                    extract_one(
                        client,
                        comment_id=cid,
                        context_prefix=rows[cid]["context_prefix"] or "",
                        text_clean=rows[cid]["text_clean"],
                    )
                    for cid in todo
                ]
            )
            _persist_batch(query_id, list(zip(todo, results)))

    # Summary
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


def _persist_batch(
    query_id: int, batch: list[tuple[int, ClaimExtraction | None]]
) -> None:
    with db.connect() as conn:
        for cid, out in batch:
            if out is None:
                conn.execute(
                    "UPDATE comments SET is_substantive = 0 WHERE id = ?", (cid,)
                )
                continue
            conn.execute(
                "UPDATE comments SET is_substantive = ? WHERE id = ?",
                (1 if out.is_substantive else 0, cid),
            )
            if not out.is_substantive:
                continue
            for cl in out.claims:
                conn.execute(
                    """
                    INSERT INTO claims (comment_id, claim_text, stance, category,
                                        evidence_type, tools_mentioned, confidence,
                                        is_firsthand)
                    VALUES (?,?,?,?,?,?,?,?)
                    """,
                    (
                        cid,
                        cl.text,
                        cl.stance,
                        cl.category,
                        cl.evidence_type,
                        json.dumps(cl.tools_mentioned, ensure_ascii=False),
                        float(cl.confidence),
                        1 if cl.is_firsthand else 0,
                    ),
                )
