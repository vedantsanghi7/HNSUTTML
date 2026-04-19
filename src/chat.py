# grounded chat with layered memory + hybrid retrieval.
#
# query router classifies intent and rewrites the query using chat memory.
# hybrid retrieval (BM25 + dense -> RRF -> re-score) pulls relevant comments.
# layered memory: pinned digest + rolling summary + last 4 verbatim turns.
# answer generation via sarvam-105b with required citations.
# groundedness verifier checks cited sentences against source comments.

from __future__ import annotations

import asyncio
import json
import logging
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src import db, retrieve
from src.config import (
    DATA_DIR,
    MODEL_LARGE,
    MODEL_SMALL,
    P5_CHAT_ANSWER_SYSTEM,
    P6_GROUNDEDNESS_JUDGE,
    P_ROLLING_SUMMARY,
    P_ROUTER_SYSTEM,
    ROUTE_QUERY_TOOL,
)
from src.extract import RouterDecision
from src.fetch import log_error
from src.llm import chat as llm_chat, extract_text, extract_tool_args

logger = logging.getLogger(__name__)

RECENT_TURNS_KEPT = 4
SUMMARY_REFRESH_EVERY = 4


# session model


@dataclass
class Turn:
    user: str
    assistant: str
    evidence_ids: list[int] = field(default_factory=list)
    intent: str = ""


@dataclass
class Session:
    session_id: str
    query_id: int
    topic: str
    digest: str
    turns: list[Turn] = field(default_factory=list)
    rolling_summary: str = ""

    def recent_pairs(self, n: int = RECENT_TURNS_KEPT) -> list[Turn]:
        return self.turns[-n:]

    def older_pairs(self, keep: int = RECENT_TURNS_KEPT) -> list[Turn]:
        if len(self.turns) <= keep:
            return []
        return self.turns[:-keep]


# session registry (in-memory, no persistence needed for a dev tool)


_SESSIONS: dict[str, Session] = {}


def _load_digest_markdown(query_id: int) -> str:
    # Prefer the cached file at data/digest_qN.md; fall back to the db-less
    # digest.md that older runs produced.
    cand = Path(DATA_DIR) / f"digest_q{query_id}.md"
    if cand.exists():
        return cand.read_text(encoding="utf-8")
    fallback = Path(DATA_DIR) / "digest.md"
    if fallback.exists():
        return fallback.read_text(encoding="utf-8")
    raise FileNotFoundError(
        f"No digest found for query_id={query_id}; expected {cand} or {fallback}"
    )


def _topic_for_query(query_id: int) -> str:
    with db.connect() as conn:
        row = conn.execute(
            "SELECT topic FROM queries WHERE id = ?", (query_id,)
        ).fetchone()
    if not row:
        raise ValueError(f"query_id {query_id} not found")
    return row["topic"]


def start_session(query_id: int) -> Session:
    topic = _topic_for_query(query_id)
    digest = _load_digest_markdown(query_id)
    sid = uuid.uuid4().hex
    sess = Session(session_id=sid, query_id=query_id, topic=topic, digest=digest)
    _SESSIONS[sid] = sess
    return sess


def get_session(session_id: str) -> Session | None:
    return _SESSIONS.get(session_id)


# router


def _compressed_context(sess: Session) -> str:
    # Minimal context for the router: rolling summary + recent pairs.
    parts: list[str] = []
    if sess.rolling_summary:
        parts.append(f"Rolling summary of older turns:\n{sess.rolling_summary}")
    recent = sess.recent_pairs()
    if recent:
        parts.append("Recent turns (most recent last):")
        for t in recent:
            parts.append(f"USER: {t.user}")
            parts.append(f"ASSISTANT: {t.assistant[:280]}")
    if not parts:
        parts.append("(no prior turns)")
    return "\n".join(parts)


async def route(sess: Session, user_msg: str) -> RouterDecision:
    # Classify + rewrite the user turn via a single Sarvam-30B tool call.
    sys = P_ROUTER_SYSTEM.format(topic=sess.topic)
    ctx = _compressed_context(sess)
    user = f"Chat context:\n{ctx}\n\nNew user message: {user_msg}"
    try:
        resp = await llm_chat(
            [
                {"role": "system", "content": sys},
                {"role": "user", "content": user},
            ],
            purpose="chat_router",
            model=MODEL_SMALL,
            tools=[ROUTE_QUERY_TOOL],
            tool_choice={"type": "function", "function": {"name": "route_query"}},
            temperature=0.1,
            max_tokens=1500,
        )
    except Exception as exc:  # noqa: BLE001
        log_error(
            "chat.route",
            f"{type(exc).__name__}: {exc}",
            traceback.format_exc(),
            inp=user_msg[:200],
        )
        return _heuristic_route(user_msg, sess)

    args = extract_tool_args(resp, "route_query")
    if not args:
        return _heuristic_route(user_msg, sess)
    try:
        return RouterDecision.model_validate(args)
    except ValidationError:
        return _heuristic_route(user_msg, sess)


def _heuristic_route(user_msg: str, sess: Session) -> RouterDecision:
    # Fallback router when the LLM call fails or returns invalid JSON.
    lower = user_msg.lower()
    anaphora = any(
        w in lower
        for w in (
            "earlier",
            "above",
            "you said",
            "you mentioned",
            "that one",
            "previously",
        )
    )
    return RouterDecision(
        intent="follow_up_reference" if anaphora else "how_to",
        rewritten_query=user_msg.strip(),
        requires_retrieval=not anaphora,
        references_earlier_turn=anaphora,
    )


# evidence formatting


def _trim(text: str, cap: int = 900) -> str:
    if len(text) <= cap:
        return text
    return text[: cap - 1].rsplit(" ", 1)[0] + "…"


def _evidence_block(rows: list[dict]) -> str:
    # Render retrieved rows as XML-tagged HN comments for the answer prompt.
    if not rows:
        return "(no retrieved evidence)"
    out: list[str] = []
    for r in rows:
        title = (r.get("thread_title") or "").replace("\n", " ").strip()
        snippet = _trim((r.get("text_clean") or "").strip())
        out.append(
            f'<hn_comment id="{r["cid"]}" thread="{title}" '
            f'points="{r.get("thread_points") or 0}" '
            f'descendants="{r.get("desc_n") or 0}">\n'
            f"{snippet}\n"
            f"</hn_comment>"
        )
    return "\n\n".join(out)


# rolling summary


async def _maybe_refresh_summary(sess: Session) -> None:
    # Regenerate the rolling summary every SUMMARY_REFRESH_EVERY turns
    # once the history exceeds the verbatim window.
    older = sess.older_pairs(keep=RECENT_TURNS_KEPT)
    if not older:
        return
    n_turns = len(sess.turns)
    if n_turns % SUMMARY_REFRESH_EVERY != 0:
        return  # cadence: refresh only every Nth turn

    old_turns = []
    for t in older:
        old_turns.append(f"USER: {t.user}\nASSISTANT: {t.assistant}")
    prompt = P_ROLLING_SUMMARY.format(
        topic=sess.topic,
        prev_summary=sess.rolling_summary or "(none)",
        old_turns="\n\n".join(old_turns),
    )
    try:
        resp = await llm_chat(
            [{"role": "user", "content": prompt}],
            purpose="chat_summary",
            model=MODEL_SMALL,
            temperature=0.2,
            max_tokens=1500,
        )
    except Exception as exc:  # noqa: BLE001
        log_error(
            "chat._maybe_refresh_summary",
            f"{type(exc).__name__}: {exc}",
            traceback.format_exc(),
        )
        return
    text = extract_text(resp).strip()
    if text:
        sess.rolling_summary = text


# answer generation


def _assemble_memory(sess: Session) -> str:
    # Tier 1 + Tier 3 + Tier 2, as a single user-visible block.
    parts: list[str] = []
    parts.append("### Pinned digest (ground truth for this topic):\n")
    parts.append(sess.digest.strip())
    if sess.rolling_summary:
        parts.append("\n### Rolling summary of older turns:\n")
        parts.append(sess.rolling_summary)
    recent = sess.recent_pairs()
    if recent:
        parts.append("\n### Recent turns (verbatim):\n")
        for t in recent:
            parts.append(f"USER: {t.user}")
            parts.append(f"ASSISTANT: {t.assistant}")
    return "\n".join(parts)


async def answer(sess: Session, user_msg: str) -> dict:
    # Run a full chat turn: route → retrieve → generate → persist.
    # Returns a dict with keys: answer, intent, rewritten_query, evidence (list),
    # citations (list[int]), used_retrieval (bool).
    decision = await route(sess, user_msg)

    # Off-topic short-circuit: no retrieval, deterministic refusal.
    if decision.intent == "off_topic":
        refusal = (
            f"The HN threads I fetched for \"{sess.topic}\" don't address this. "
            f"I can only answer follow-ups grounded in those threads."
        )
        sess.turns.append(Turn(user=user_msg, assistant=refusal, intent="off_topic"))
        await _maybe_refresh_summary(sess)
        return {
            "answer": refusal,
            "intent": "off_topic",
            "rewritten_query": decision.rewritten_query,
            "evidence": [],
            "citations": [],
            "used_retrieval": False,
        }

    evidence_rows: list[dict] = []
    if decision.requires_retrieval:
        evidence_rows = retrieve.hybrid_retrieve(
            decision.rewritten_query, sess.query_id
        )

    # Pure follow-up with no new evidence: let the model lean on memory; if we
    # actually have no retrieved rows we still include the digest + recent turns
    # as grounding, which is enough for anaphora resolution.
    evidence_block = _evidence_block(evidence_rows)
    memory_block = _assemble_memory(sess)
    sys = P5_CHAT_ANSWER_SYSTEM.format(topic=sess.topic)
    user = (
        f"=== MEMORY ===\n{memory_block}\n\n"
        f"=== RETRIEVED EVIDENCE (for this turn only) ===\n{evidence_block}\n\n"
        f"=== USER MESSAGE ===\n{user_msg}"
    )
    try:
        resp = await llm_chat(
            [
                {"role": "system", "content": sys},
                {"role": "user", "content": user},
            ],
            purpose="chat_answer",
            model=MODEL_LARGE,
            temperature=0.3,
            max_tokens=2500,
        )
    except Exception as exc:  # noqa: BLE001
        log_error(
            "chat.answer",
            f"{type(exc).__name__}: {exc}",
            traceback.format_exc(),
            inp=user_msg[:200],
        )
        fallback = (
            "I'm temporarily unable to reach the language model. "
            "Please retry the question in a moment."
        )
        sess.turns.append(Turn(user=user_msg, assistant=fallback, intent=decision.intent))
        return {
            "answer": fallback,
            "intent": decision.intent,
            "rewritten_query": decision.rewritten_query,
            "evidence": [],
            "citations": [],
            "used_retrieval": False,
        }

    text = extract_text(resp)

    is_grounded = await _verify_groundedness(text, evidence_rows)
    if not is_grounded:
        user_retry = user + "\n\nCRITICAL REMINDER: Your previous answer contained claims NOT supported by the cited evidence. Please rewrite and ensure EVERY claim is strictly supported by the provided <hn_comment>s."
        try:
            resp2 = await llm_chat(
                [
                    {"role": "system", "content": sys},
                    {"role": "user", "content": user_retry},
                ],
                purpose="chat_answer_retry",
                model=MODEL_LARGE,
                temperature=0.3,
                max_tokens=2500,
            )
            text2 = extract_text(resp2)
            is_grounded2 = await _verify_groundedness(text2, evidence_rows)
            if not is_grounded2:
                text = "I couldn't find solid evidence for one part of this. The supported claims are:\n\n" + text2
            else:
                text = text2
        except Exception as exc:
            log_error("chat.answer_retry", f"{type(exc).__name__}: {exc}", traceback.format_exc())
            # fall back to original text if retry fails purely due to API error

    citations = _extract_cite_ids(text)

    evidence_ids = [r["cid"] for r in evidence_rows]
    sess.turns.append(
        Turn(
            user=user_msg,
            assistant=text,
            evidence_ids=evidence_ids,
            intent=decision.intent,
        )
    )
    await _maybe_refresh_summary(sess)

    # Slim evidence for the response payload.
    evidence_out = [
        {
            "cid": r["cid"],
            "thread_title": r.get("thread_title"),
            "thread_points": r.get("thread_points"),
            "snippet": _trim(r.get("text_clean") or "", 240),
        }
        for r in evidence_rows
    ]
    return {
        "answer": text,
        "intent": decision.intent,
        "rewritten_query": decision.rewritten_query,
        "evidence": evidence_out,
        "citations": citations,
        "used_retrieval": decision.requires_retrieval and bool(evidence_rows),
    }


# helpers


import re as _re

_CITE_RE = _re.compile(r"\[#([0-9][^\]]*)\]")


def _extract_cite_ids(text: str) -> list[int]:
    ids: list[int] = []
    for m in _CITE_RE.finditer(text or ""):
        for num in _re.findall(r"(\d+)", m.group(1)):
            ids.append(int(num))
    return ids


async def _verify_groundedness(text: str, evidence_rows: list[dict]) -> bool:
    # Returns True if grounded, False if any sentence is UNSUPPORTED.
    ev_map = {r["cid"]: r.get("text_clean") or "" for r in evidence_rows}
    # Naive sentence split by common punctuation followed by space
    sentences = _re.split(r'(?<=[.!?])\s+', text)
    tasks = []
    
    for sentence in sentences:
        cites = _extract_cite_ids(sentence)
        for cid in cites:
            if cid in ev_map:
                tasks.append(_check_one(sentence, cid, ev_map[cid]))
                
    if not tasks:
        return True
        
    results = await asyncio.gather(*tasks)
    return not any("UNSUPPORTED" in r for r in results)


async def _check_one(sentence: str, cid: int, full_text: str) -> str:
    prompt = P6_GROUNDEDNESS_JUDGE.format(sentence=sentence, id=cid, full_text=full_text)
    try:
        resp = await llm_chat(
            [{"role": "user", "content": prompt}],
            purpose="chat_verify",
            model=MODEL_SMALL,
            temperature=0.0,
            max_tokens=10,
        )
        return extract_text(resp).strip().upper()
    except Exception:
        # If API fails for one check, assume PARTIAL rather than crashing or blocking
        return "PARTIAL"
