# digest synthesis + citation verification.
#
# builds cluster summary blocks, detects contradictions between clusters,
# synthesizes via sarvam-105b, and verifies that every [#id] citation
# actually exists in the query's comment set.

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

import logging

from src import db
from src.config import MODEL_LARGE, P4_DIGEST_SYNTHESIS_SYSTEM
from src.fetch import log_error
from src.llm import chat, extract_text

logger = logging.getLogger(__name__)

CITATION_RE = re.compile(r"\[#(\d+)(?:,\s*#?(\d+))*\]")
ID_IN_CITE_RE = re.compile(r"#?(\d+)")

# tool names to exclude from the alternatives table since they're the
# subject of the digest, not alternatives.
_SELF_TOOL_TOKENS = {"sqlite", "sqlite3", "sqlite-3"}


def _topic_tokens(topic: str) -> list[str]:
    # Tokens from the topic used for on-topic cluster filtering.
    # Also splits camelCase / mixed-case compounds so e.g. 'macOS' produces
    # both 'macos' and 'mac', and 'PostgreSQL' produces 'postgresql' and 'postgres'.
    raw_original = re.findall(r"[A-Za-z][A-Za-z0-9+]{2,}", topic)
    stop = {"the", "and", "for", "with", "that", "this", "from", "into", "its",
            "are", "was", "been", "has", "have", "not", "but", "about", "use",
            "how", "why", "what", "can", "should", "does", "will"}
    tokens = [w.lower() for w in raw_original if w.lower() not in stop]
    # Also add sub-word splits for camelCase or mixed-case words
    extra: list[str] = []
    for original in raw_original:
        # Split on case transitions using the ORIGINAL casing:
        # 'macOS' → ['mac', 'OS'], 'PostgreSQL' → ['Postgre', 'SQL']
        parts = re.findall(r"[a-z]+|[A-Z][a-z]*|[0-9]+", original)
        for p in parts:
            p = p.lower()
            if len(p) >= 3 and p not in stop and p not in tokens:
                extra.append(p)
    tokens.extend(extra)
    return tokens


def _thread_titles_by_id(query_id: int) -> dict[int, str]:
    with db.connect() as conn:
        return {
            r["id"]: (r["title"] or "")
            for r in conn.execute(
                "SELECT id, title FROM threads WHERE query_id = ?", (query_id,)
            ).fetchall()
        }


def _score_cluster_relevance(
    cluster: dict, thread_titles: dict[int, str], topic_toks: list[str]
) -> float:
    # Return the fraction of cluster members whose thread title contains a topic token.
    total = 0
    on_topic = 0
    for m in cluster["members"]:
        title = thread_titles.get(m["thread_id"], "").lower()
        total += 1
        if any(tok in title for tok in topic_toks):
            on_topic += 1
    return on_topic / total if total else 1.0


def _filter_off_topic(
    clusters: list[dict], thread_titles: dict[int, str], topic_toks: list[str]
) -> list[dict]:
    # Filter clusters that seem off-topic with graceful fallback.
    # Uses cascading thresholds: 50% → 30% → keep all.
    # Never returns an empty list if the input was non-empty.
    if not topic_toks or not clusters:
        return clusters

    # Score every cluster once
    scored = [(c, _score_cluster_relevance(c, thread_titles, topic_toks)) for c in clusters]

    # Try strict threshold first (50%), then relaxed (30%)
    for threshold in (0.5, 0.3):
        kept = [c for c, score in scored if score >= threshold]
        if kept:
            if len(kept) < len(clusters):
                logger.info(
                    "Off-topic filter (threshold=%.0f%%): kept %d / %d clusters",
                    threshold * 100, len(kept), len(clusters),
                )
            return kept

    # If nothing passes even a 30% threshold, keep everything with a warning.
    # This prevents crashes when search returns tangentially-related threads.
    logger.warning(
        "Off-topic filter: no clusters passed any threshold for topic tokens %s; "
        "keeping all %d clusters. The search likely returned tangential threads.",
        topic_toks, len(clusters),
    )
    return clusters


def _compile_alternatives(clusters: list[dict]) -> list[dict]:
    # Aggregate tools mentioned across cluster claims into a ranked table
    # with representative comment ids.
    by_tool: dict[str, dict] = {}
    for c in clusters:
        for m in c["members"]:
            try:
                tools = json.loads(m.get("tools_mentioned") or "[]")
            except Exception:  # noqa: BLE001
                continue
            for raw in tools:
                if not raw:
                    continue
                tool = raw.strip()
                key = tool.lower()
                if key in _SELF_TOOL_TOKENS:
                    continue
                entry = by_tool.setdefault(
                    key,
                    {
                        "display": tool,
                        "count": 0,
                        "examples": [],
                    },
                )
                entry["count"] += 1
                if len(entry["examples"]) < 3:
                    entry["examples"].append(
                        {
                            "comment_id": m["comment_id"],
                            "claim": m["claim_text"],
                        }
                    )
    rows = sorted(by_tool.values(), key=lambda r: r["count"], reverse=True)
    return rows


def _firsthand_war_stories(clusters: list[dict], limit: int = 6) -> list[dict]:
    stories = []
    for c in clusters:
        for m in c["members"]:
            if m["is_firsthand"] and m["stance"] in ("anecdote", "benchmark", "pro", "con"):
                stories.append(m)
    stories.sort(key=lambda m: m["confidence"], reverse=True)
    seen = set()
    out = []
    for s in stories:
        if s["comment_id"] in seen:
            continue
        seen.add(s["comment_id"])
        out.append(s)
        if len(out) >= limit:
            break
    return out


def _load_clusters(query_id: int) -> list[dict]:
    with db.connect() as conn:
        rows = conn.execute(
            """
            SELECT id, label, stance, category, weight, member_claim_ids
            FROM clusters WHERE query_id = ?
            ORDER BY weight DESC
            """,
            (query_id,),
        ).fetchall()
        clusters = [dict(r) for r in rows]

        # Hydrate members with claim + comment info.
        for c in clusters:
            ids = json.loads(c["member_claim_ids"])
            if not ids:
                c["members"] = []
                continue
            mrows = conn.execute(
                """
                SELECT cl.id, cl.comment_id, cl.claim_text, cl.stance, cl.category,
                       cl.evidence_type, cl.confidence, cl.is_firsthand,
                       cl.tools_mentioned,
                       cmt.thread_id, th.points AS thread_points
                FROM claims cl
                JOIN comments cmt ON cmt.id = cl.comment_id
                JOIN threads th ON th.id = cmt.thread_id
                WHERE cl.id IN ({})
                """.format(",".join("?" * len(ids))),
                ids,
            ).fetchall()
            c["members"] = [dict(r) for r in mrows]
    return clusters


def _detect_debates(clusters: list[dict]) -> list[dict]:
    opposing = [("pro", "con"), ("pro", "alternative")]
    debates: list[dict] = []
    for i, a in enumerate(clusters):
        for b in clusters[i + 1 :]:
            if a["category"] != b["category"]:
                continue
            pair = (a["stance"], b["stance"])
            if pair in opposing or pair[::-1] in opposing:
                debates.append({"a": a, "b": b})
    return debates


def _build_input(
    topic: str,
    clusters: list[dict],
    debates: list[dict],
    alternatives: list[dict],
    war_stories: list[dict],
) -> str:
    # Serialize clusters + pre-computed tables into a structured prompt payload.
    lines: list[str] = []
    lines.append(f"Topic: {topic}\n")
    lines.append("## Clusters (ordered by weight)\n")
    for i, c in enumerate(clusters, 1):
        lines.append(
            f"### Cluster {i}: {c['label']}\n"
            f"- stance: {c['stance']}; category: {c['category']}; weight: {c['weight']}\n"
            f"- members: {len(c['members'])} claims from comments {[m['comment_id'] for m in c['members']]}\n"
        )
        ms = sorted(c["members"], key=lambda m: m["confidence"], reverse=True)
        for m in ms[:10]:
            tools = ""
            try:
                tlist = json.loads(m.get("tools_mentioned") or "[]")
                if tlist:
                    tools = f" tools={tlist}"
            except Exception:  # noqa: BLE001
                pass
            lines.append(
                f'  - "{m["claim_text"]}" '
                f'[stance={m["stance"]}, ev={m["evidence_type"]}, '
                f'firsthand={bool(m["is_firsthand"])}, conf={m["confidence"]:.2f}{tools}] '
                f'[#{m["comment_id"]}]'
            )

    if debates:
        lines.append("\n## Detected debates (opposing stances in same category)")
        for d in debates:
            lines.append(
                f"- **{d['a']['category']}**: "
                f"{d['a']['stance']} ({d['a']['label']}) vs "
                f"{d['b']['stance']} ({d['b']['label']})"
            )

    if alternatives:
        lines.append("\n## Pre-computed alternatives (fill the Alternatives Mentioned table from this)")
        for a in alternatives[:15]:
            sample = a["examples"][0] if a["examples"] else None
            ex = (
                f'e.g. [#{sample["comment_id"]}]: "{sample["claim"][:140]}"'
                if sample
                else ""
            )
            lines.append(f"- **{a['display']}** - mentions: {a['count']}. {ex}")

    if war_stories:
        lines.append("\n## Firsthand war-story candidates (pick the strongest for Notable War Stories)")
        for s in war_stories:
            lines.append(
                f'- [#{s["comment_id"]}] "{s["claim_text"][:180]}" '
                f'(stance={s["stance"]}, ev={s["evidence_type"]}, conf={s["confidence"]:.2f})'
            )

    return "\n".join(lines)


def _all_member_ids(clusters: list[dict]) -> set[int]:
    ids: set[int] = set()
    for c in clusters:
        for m in c["members"]:
            ids.add(int(m["comment_id"]))
    return ids


def _extract_cites(text: str) -> list[int]:
    # Find every numeric id inside [#...] brackets (supports `[#1,#2]` style).
    out: list[int] = []
    for match in re.finditer(r"\[#([0-9][^\]]*)\]", text):
        inside = match.group(1)
        for num in re.findall(r"(\d+)", inside):
            out.append(int(num))
    return out


def _all_bullets_have_cite(text: str) -> bool:
    # Every top-level bullet line should contain at least one [#id].
    for ln in text.splitlines():
        s = ln.strip()
        if not s.startswith("- "):
            continue
        # skip table rows and War Stories '  - [#id] "..."' (already cited)
        if "[#" not in s:
            return False
    return True


async def synthesize(query_id: int) -> str:
    with db.connect() as conn:
        q = conn.execute(
            "SELECT topic FROM queries WHERE id = ?", (query_id,)
        ).fetchone()
        topic = q["topic"]

    clusters = _load_clusters(query_id)
    if not clusters:
        raise RuntimeError("No clusters found. Run extract + cluster first.")

    titles = _thread_titles_by_id(query_id)
    topic_toks = _topic_tokens(topic)
    clusters = _filter_off_topic(clusters, titles, topic_toks)

    debates = _detect_debates(clusters)
    alternatives = _compile_alternatives(clusters)
    war_stories = _firsthand_war_stories(clusters)

    payload = _build_input(topic, clusters, debates, alternatives, war_stories)
    valid_ids = _all_member_ids(clusters)

    async def _call(reinforce: bool = False) -> str:
        sys_msg = P4_DIGEST_SYNTHESIS_SYSTEM.format(topic=topic)
        sys_msg += (
            "\n\nADDITIONAL GUIDANCE:\n"
            "- Aim for 700-1200 words. Use specific numbers, tool names, versions, quoted phrases.\n"
            "- Fill every section of the template. If a section truly has nothing, write '(none found)'.\n"
            "- The Alternatives Mentioned table MUST have at least 3 rows if pre-computed alternatives were provided.\n"
            "- The Notable War Stories section should list 3-5 firsthand anecdotes from the provided candidates."
        )
        if reinforce:
            sys_msg += (
                "\n\nREINFORCE: Your previous attempt cited comment ids not in the clusters. "
                "Every [#id] MUST be one of the ids listed in the Members of some cluster above. "
                "Do not invent ids."
            )
        resp = await chat(
            [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": payload},
            ],
            purpose="digest_synthesis",
            model=MODEL_LARGE,
            temperature=0.3,
            max_tokens=4000,
        )
        return extract_text(resp)

    text = await _call()
    cited = _extract_cites(text)
    bad = [c for c in cited if c not in valid_ids]
    if bad:
        log_error(
            "digest.synthesize",
            f"{len(bad)} citations not in cluster members; regenerating",
            "",
            inp=f"bad_ids={bad[:10]}",
        )
        text = await _call(reinforce=True)
        cited = _extract_cites(text)
        bad = [c for c in cited if c not in valid_ids]
        if bad:
            log_error(
                "digest.synthesize",
                f"still {len(bad)} bad citations after retry; returning anyway",
                "",
                inp=f"bad_ids={bad[:10]}",
            )
    return text
