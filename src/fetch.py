# multi-axis HN fetch, normalize, persist, and audit.
# handles the algolia search fan-out, firebase comment tree BFS,
# HTML normalization, and the data quality audit report.

from __future__ import annotations

import asyncio
import html
import json
import re
import time
import traceback
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import httpx
from bs4 import BeautifulSoup
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from src import db
from src.config import (
    ALGOLIA_SEARCH_BY_DATE_URL,
    ALGOLIA_SEARCH_URL,
    AUDIT_REPORT_PATH,
    ERROR_LOG_PATH,
    FETCH_IDEMPOTENCY_WINDOW_SEC,
    FIREBASE_ITEM_URL,
    HN_CONCURRENCY,
    HN_FETCH_RETRY_MAX,
    HN_MAX_DEPTH,
    MIN_NUM_COMMENTS,
    MIN_TEXT_LENGTH_NO_CODE,
    POINTS_MIN_ALL_TIME,
    POINTS_MIN_RECENT,
    RECENCY_WINDOW_DAYS,
    SLOT_TARGETS,
)

# error logging


def log_error(where: str, what: str, tb: str, inp: str = "") -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    entry = (
        f"\n## [{ts}] {where}\n\n"
        f"**What failed:** {what}\n\n"
        f"**Input:** {inp}\n\n"
        f"**Traceback:**\n```\n{tb}\n```\n"
    )
    with open(ERROR_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(entry)


# algolia search (multi-axis fan-out)


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return code == 429 or 500 <= code < 600
    return isinstance(exc, (httpx.RequestError, httpx.TimeoutException))


@retry(
    retry=retry_if_exception(_is_transient),
    wait=wait_exponential_jitter(initial=1, max=30),
    stop=stop_after_attempt(HN_FETCH_RETRY_MAX),
    reraise=True,
)
async def _http_get_json(client: httpx.AsyncClient, url: str, params: dict | None = None) -> dict:
    r = await client.get(url, params=params, timeout=20.0)
    r.raise_for_status()
    return r.json()


async def _algolia_hits(
    client: httpx.AsyncClient, *, base_url: str, params: dict
) -> list[dict]:
    data = await _http_get_json(client, base_url, params=params)
    return data.get("hits", [])


def _story_filter(hit: dict) -> bool:
    if not hit.get("objectID"):
        return False
    if (hit.get("num_comments") or 0) < MIN_NUM_COMMENTS:
        return False
    # Algolia's story hits don't have dead/deleted flags but some edge cases exist.
    if hit.get("dead") or hit.get("deleted"):
        return False
    return True


async def search_slots(client: httpx.AsyncClient, topic: str) -> list[tuple[str, dict]]:
    # Return (slot_name, algolia_hit) pairs, deduped by story id across slots.
    now = int(time.time())
    recent_cutoff = now - RECENCY_WINDOW_DAYS * 86400

    # Fetch wide then trim/sort client-side (Algolia can't sort-by-points on /search).
    relevance_task = _algolia_hits(
        client,
        base_url=ALGOLIA_SEARCH_URL,
        params={"query": topic, "tags": "story", "hitsPerPage": 30},
    )
    points_task = _algolia_hits(
        client,
        base_url=ALGOLIA_SEARCH_URL,
        params={
            "query": topic,
            "tags": "story",
            "numericFilters": f"points>{POINTS_MIN_ALL_TIME}",
            "hitsPerPage": 50,
        },
    )
    recency_task = _algolia_hits(
        client,
        base_url=ALGOLIA_SEARCH_BY_DATE_URL,
        params={
            "query": topic,
            "tags": "story",
            "numericFilters": f"points>{POINTS_MIN_RECENT},created_at_i>{recent_cutoff}",
            "hitsPerPage": 30,
        },
    )
    ask_task = _algolia_hits(
        client,
        base_url=ALGOLIA_SEARCH_URL,
        params={"query": topic, "tags": "(ask_hn,show_hn)", "hitsPerPage": 20},
    )

    relevance, points, recency, ask = await asyncio.gather(
        relevance_task, points_task, recency_task, ask_task
    )

    relevance = [h for h in relevance if _story_filter(h)]
    points = sorted(
        [h for h in points if _story_filter(h)],
        key=lambda h: h.get("points") or 0,
        reverse=True,
    )
    recency = [h for h in recency if _story_filter(h)]  # already date-sorted
    ask = [h for h in ask if _story_filter(h)]

    picked: list[tuple[str, dict]] = []
    seen: set[str] = set()

    def _take(slot: str, hits: Iterable[dict], n: int) -> None:
        count = 0
        for h in hits:
            sid = h["objectID"]
            if sid in seen:
                continue
            picked.append((slot, h))
            seen.add(sid)
            count += 1
            if count >= n:
                return

    _take("relevance", relevance, SLOT_TARGETS["relevance"])
    _take("points", points, SLOT_TARGETS["points"])
    _take("recency", recency, SLOT_TARGETS["recency"])
    _take("ask_hn", ask, SLOT_TARGETS["ask_hn"])

    return picked


# firebase comment-tree BFS


@dataclass
class CommentNode:
    id: int
    parent_id: int | None
    author: str | None
    created_at: int | None
    text_raw: str
    depth: int
    kids: list[int]
    dead: bool
    deleted: bool


async def _fetch_item(client: httpx.AsyncClient, item_id: int) -> dict | None:
    try:
        data = await _http_get_json(client, FIREBASE_ITEM_URL.format(id=item_id))
    except Exception as exc:  # noqa: BLE001
        log_error(
            "fetch._fetch_item",
            f"failed after retries: {type(exc).__name__}: {exc}",
            traceback.format_exc(),
            inp=f"item_id={item_id}",
        )
        return None
    return data


async def fetch_comment_tree(
    client: httpx.AsyncClient, root_id: int, sem: asyncio.Semaphore
) -> list[CommentNode]:
    # BFS by depth, each level concurrent under the semaphore.

    async def _bounded_fetch(iid: int) -> dict | None:
        async with sem:
            return await _fetch_item(client, iid)

    # Level 0 = story (its kids are top-level comments, depth=0).
    story = await _bounded_fetch(root_id)
    if not story:
        return []

    nodes: list[CommentNode] = []
    frontier: list[tuple[int, int, int | None]] = [
        (kid, 0, None) for kid in story.get("kids", []) or []
    ]  # (item_id, depth, parent_id_for_comment)

    while frontier and frontier[0][1] <= HN_MAX_DEPTH:
        depth = frontier[0][1]
        current_level = [t for t in frontier if t[1] == depth]
        frontier = [t for t in frontier if t[1] != depth]

        items = await asyncio.gather(*(_bounded_fetch(t[0]) for t in current_level))

        for (iid, d, pid), item in zip(current_level, items):
            if item is None:
                continue
            if item.get("type") != "comment":
                continue
            kids = item.get("kids") or []
            node = CommentNode(
                id=item["id"],
                parent_id=pid if pid is not None else root_id,  # top-level points at story
                author=item.get("by"),
                created_at=item.get("time"),
                text_raw=item.get("text") or "",
                depth=d,
                kids=list(kids),
                dead=bool(item.get("dead")),
                deleted=bool(item.get("deleted")),
            )
            # skip dead/deleted comments but still recurse into their kids
            nodes.append(node)
            if d < HN_MAX_DEPTH:
                for kid in kids:
                    frontier.append((kid, d + 1, node.id))

    return nodes


# normalization

_QUOTE_LINE_RE = re.compile(r"^\s*>", re.MULTILINE)


def normalize_comment_html(raw: str) -> tuple[str, bool, float]:
    # Return (text_clean, has_code, quote_density).
    # - <p> -> paragraph break
    # - <pre>/<code> preserved as fenced code
    # - entities decoded
    # - other tags stripped
    if not raw:
        return "", False, 0.0

    soup = BeautifulSoup(raw, "lxml")

    has_code = False
    # Preserve <pre> blocks first (may contain <code>).
    for pre in soup.find_all("pre"):
        code_text = pre.get_text()
        fenced = f"\n\n```\n{code_text.strip()}\n```\n\n"
        pre.replace_with(fenced)
        has_code = True
    # Inline <code>
    for c in soup.find_all("code"):
        t = c.get_text()
        c.replace_with(f"`{t}`")
        has_code = True
    # <p> -> double newline
    for p in soup.find_all("p"):
        p.insert_before("\n\n")
        p.unwrap()
    # <a href> -> "text (href)" when different
    for a in soup.find_all("a"):
        text = a.get_text()
        href = a.get("href", "")
        if href and href != text:
            a.replace_with(f"{text} ({href})")
        else:
            a.replace_with(text)
    # <i>, <b> -> just unwrap
    for tag in soup.find_all(True):
        tag.unwrap()

    text = soup.get_text()
    text = html.unescape(text)
    # Collapse excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    # quote_density: fraction of non-empty lines starting with '>'
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        quote_density = 0.0
    else:
        q = sum(1 for ln in lines if _QUOTE_LINE_RE.match(ln))
        quote_density = q / len(lines)

    return text, has_code, quote_density


def _compute_descendant_counts(nodes: list[CommentNode]) -> dict[int, int]:
    # Count descendants per comment using parent pointers within this thread.
    children: dict[int, list[int]] = {}
    for n in nodes:
        children.setdefault(n.parent_id or -1, []).append(n.id)

    counts: dict[int, int] = {}

    def _count(node_id: int) -> int:
        kids = children.get(node_id, [])
        total = 0
        for k in kids:
            total += 1 + _count(k)
        counts[node_id] = total
        return total

    for n in nodes:
        if n.id not in counts:
            _count(n.id)
    return counts


# persistence


def _recent_query_id(conn, topic: str) -> int | None:
    cutoff = int(time.time()) - FETCH_IDEMPOTENCY_WINDOW_SEC
    row = conn.execute(
        "SELECT id FROM queries WHERE topic = ? AND fetched_at >= ? ORDER BY fetched_at DESC LIMIT 1",
        (topic, cutoff),
    ).fetchone()
    return row["id"] if row else None


def _persist_threads_and_comments(
    conn,
    *,
    query_id: int,
    picked: list[tuple[str, dict]],
    trees: list[list[CommentNode]],
) -> tuple[int, int, int]:
    # Return (threads_written, comments_written, comments_kept).
    threads_written = 0
    comments_written = 0
    comments_kept = 0

    for (slot, hit), nodes in zip(picked, trees):
        story_id = int(hit["objectID"])
        # Upsert thread
        conn.execute(
            """
            INSERT OR REPLACE INTO threads
                (id, query_id, title, url, points, num_comments, created_at, author, slot, raw_json)
            VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            (
                story_id,
                query_id,
                hit.get("title") or "",
                hit.get("url"),
                hit.get("points") or 0,
                hit.get("num_comments") or 0,
                hit.get("created_at_i") or 0,
                hit.get("author"),
                slot,
                json.dumps(hit, ensure_ascii=False),
            ),
        )
        threads_written += 1

        # Dedupe comments within a thread by cleaned-text identity
        seen_clean: set[str] = set()
        descendants = _compute_descendant_counts(nodes)

        for node in nodes:
            text_clean, has_code, quote_density = normalize_comment_html(node.text_raw)
            text_length = len(text_clean)

            discarded = 0
            reason: str | None = None

            if node.dead or node.deleted:
                if not node.kids:
                    discarded = 1
                    reason = "dead_or_deleted_no_kids"
                else:
                    discarded = 1
                    reason = "dead_or_deleted_with_kids"
            elif text_length < MIN_TEXT_LENGTH_NO_CODE and not has_code:
                discarded = 1
                reason = "too_short_no_code"
            elif text_clean and text_clean in seen_clean:
                discarded = 1
                reason = "duplicate_within_thread"
            else:
                seen_clean.add(text_clean)

            conn.execute(
                """
                INSERT OR REPLACE INTO comments
                    (id, thread_id, parent_id, author, created_at, text, text_clean,
                     depth, descendant_count, text_length, has_code, quote_density,
                     context_prefix, is_substantive, discarded, discard_reason)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,NULL,NULL,?,?)
                """,
                (
                    node.id,
                    story_id,
                    node.parent_id,
                    node.author,
                    node.created_at,
                    node.text_raw,
                    text_clean,
                    node.depth,
                    descendants.get(node.id, 0),
                    text_length,
                    1 if has_code else 0,
                    float(quote_density),
                    discarded,
                    reason,
                ),
            )
            comments_written += 1
            if not discarded:
                comments_kept += 1

    return threads_written, comments_written, comments_kept


# public entry point


async def run_fetch(topic: str) -> dict[str, Any]:
    # Fetch, normalize, persist. Return a summary dict. Idempotent within window.
    db.init_db()

    with db.connect() as conn:
        existing = _recent_query_id(conn, topic)
        if existing is not None:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM threads WHERE query_id = ?", (existing,)
            ).fetchone()
            kept = conn.execute(
                """
                SELECT COUNT(*) AS n FROM comments c
                JOIN threads t ON t.id = c.thread_id
                WHERE t.query_id = ? AND c.discarded = 0
                """,
                (existing,),
            ).fetchone()
            total = conn.execute(
                """
                SELECT COUNT(*) AS n FROM comments c
                JOIN threads t ON t.id = c.thread_id
                WHERE t.query_id = ?
                """,
                (existing,),
            ).fetchone()
            return {
                "query_id": existing,
                "cached": True,
                "threads": row["n"],
                "comments_total": total["n"],
                "comments_kept": kept["n"],
            }

    sem = asyncio.Semaphore(HN_CONCURRENCY)
    async with httpx.AsyncClient(headers={"User-Agent": "hn-intel/0.1"}) as client:
        picked = await search_slots(client, topic)
        if not picked:
            raise RuntimeError(f"No stories found for topic: {topic!r}")

        trees = await asyncio.gather(
            *(fetch_comment_tree(client, int(h["objectID"]), sem) for _, h in picked)
        )

    with db.connect() as conn:
        cur = conn.execute(
            "INSERT INTO queries (topic, fetched_at, config_json) VALUES (?,?,?)",
            (
                topic,
                int(time.time()),
                json.dumps(
                    {
                        "slot_targets": SLOT_TARGETS,
                        "min_num_comments": MIN_NUM_COMMENTS,
                        "max_depth": HN_MAX_DEPTH,
                    }
                ),
            ),
        )
        query_id = cur.lastrowid

        threads_written, comments_written, comments_kept = _persist_threads_and_comments(
            conn, query_id=query_id, picked=picked, trees=trees
        )

    return {
        "query_id": query_id,
        "cached": False,
        "threads": threads_written,
        "comments_total": comments_written,
        "comments_kept": comments_kept,
        "picked_slots": Counter(slot for slot, _ in picked),
    }


# audit report


def _histogram(values: list[int | float], bins: int = 10, width: int = 40) -> str:
    if not values:
        return "(no data)"
    lo, hi = min(values), max(values)
    if lo == hi:
        return f"{lo:>8.0f} | {'#' * width} ({len(values)})"
    step = (hi - lo) / bins
    buckets = [0] * bins
    for v in values:
        idx = min(int((v - lo) / step), bins - 1)
        buckets[idx] += 1
    peak = max(buckets) or 1
    lines = []
    for i, c in enumerate(buckets):
        b_lo = lo + i * step
        bar = "#" * int(width * c / peak)
        lines.append(f"{b_lo:>8.1f} | {bar} {c}")
    return "\n".join(lines)


def write_audit(query_id: int, out_path: Path = AUDIT_REPORT_PATH) -> str:
    # Generate the data audit report.
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with db.connect() as conn:
        q = conn.execute(
            "SELECT topic, fetched_at FROM queries WHERE id = ?", (query_id,)
        ).fetchone()
        if not q:
            raise ValueError(f"No query with id {query_id}")
        topic = q["topic"]
        fetched_at = q["fetched_at"]

        threads = conn.execute(
            "SELECT id, title, slot, points, num_comments FROM threads WHERE query_id = ? ORDER BY slot, points DESC",
            (query_id,),
        ).fetchall()

        total = conn.execute(
            "SELECT COUNT(*) AS n FROM comments c JOIN threads t ON t.id = c.thread_id WHERE t.query_id = ?",
            (query_id,),
        ).fetchone()["n"]
        kept = conn.execute(
            "SELECT COUNT(*) AS n FROM comments c JOIN threads t ON t.id = c.thread_id WHERE t.query_id = ? AND c.discarded = 0",
            (query_id,),
        ).fetchone()["n"]
        discarded = total - kept

        depths = [
            r["depth"]
            for r in conn.execute(
                "SELECT depth FROM comments c JOIN threads t ON t.id = c.thread_id WHERE t.query_id = ? AND c.discarded = 0",
                (query_id,),
            ).fetchall()
        ]
        lengths = [
            r["text_length"]
            for r in conn.execute(
                "SELECT text_length FROM comments c JOIN threads t ON t.id = c.thread_id WHERE t.query_id = ? AND c.discarded = 0",
                (query_id,),
            ).fetchall()
        ]
        points_list = [r["points"] or 0 for r in threads]

        reasons = Counter(
            r["discard_reason"]
            for r in conn.execute(
                "SELECT discard_reason FROM comments c JOIN threads t ON t.id = c.thread_id WHERE t.query_id = ? AND c.discarded = 1",
                (query_id,),
            ).fetchall()
        )

        per_thread = conn.execute(
            """
            SELECT t.id, t.title, t.slot, t.points, t.num_comments,
                   SUM(CASE WHEN c.discarded = 0 THEN 1 ELSE 0 END) AS kept,
                   SUM(CASE WHEN c.discarded = 1 THEN 1 ELSE 0 END) AS dropped
            FROM threads t LEFT JOIN comments c ON c.thread_id = t.id
            WHERE t.query_id = ?
            GROUP BY t.id
            ORDER BY t.slot, t.points DESC
            """,
            (query_id,),
        ).fetchall()

        slot_spread = sorted({t["slot"] for t in threads})

    # commentary - deterministic, data-driven, no LLM involved
    short_count = reasons.get("too_short_no_code", 0)
    dup_count = reasons.get("duplicate_within_thread", 0)
    dead_nokids = reasons.get("dead_or_deleted_no_kids", 0)
    avg_len = (sum(lengths) / len(lengths)) if lengths else 0
    median_depth = sorted(depths)[len(depths) // 2] if depths else 0

    commentary = (
        f"The corpus spans {len(threads)} threads across {len(slot_spread)} retrieval slots "
        f"({', '.join(slot_spread)}). After filtering we kept {kept}/{total} comments "
        f"(avg length {avg_len:.0f} chars, median depth {median_depth}). "
        f"The most common discard reason was "
        f"{'short non-code replies' if short_count >= dup_count else 'within-thread duplicates'} "
        f"({short_count} short, {dup_count} duplicates, {dead_nokids} dead-without-kids). "
        "Short-but-code-bearing comments were deliberately kept because code snippets often "
        "carry the most concrete evidence even when prose is sparse. Throwaway-account comments "
        "were kept too -- they're often the most candid. Dead/deleted comments "
        "with surviving children were pruned at the node itself but the children remain; this "
        "preserves the thread's argumentative shape without propagating noise."
    )

    # Assemble markdown
    when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(fetched_at))

    lines: list[str] = []
    lines.append(f"# Audit: {topic}")
    lines.append("")
    lines.append(f"_Query id: {query_id} • Fetched: {when}_")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Threads: **{len(threads)}** across {len(slot_spread)} slots")
    lines.append(f"- Comments fetched: **{total}**")
    lines.append(f"- Comments kept: **{kept}** ({(kept/total*100 if total else 0):.1f}%)")
    lines.append(f"- Comments discarded: **{discarded}**")
    lines.append("")
    lines.append("## Threads")
    lines.append("")
    lines.append("| Slot | Title | Points | HN comments | Kept | Dropped |")
    lines.append("|---|---|---|---|---|---|")
    for r in per_thread:
        title = (r["title"] or "").replace("|", "\\|")
        if len(title) > 80:
            title = title[:77] + "..."
        lines.append(
            f"| {r['slot']} | {title} | {r['points']} | {r['num_comments']} | {r['kept'] or 0} | {r['dropped'] or 0} |"
        )
    lines.append("")
    lines.append("## Discard reasons")
    lines.append("")
    if reasons:
        for reason, n in reasons.most_common():
            lines.append(f"- `{reason or '(none)'}`: {n}")
    else:
        lines.append("(none)")
    lines.append("")
    lines.append("## Histograms")
    lines.append("")
    lines.append("### Comment depth (kept)")
    lines.append("```")
    lines.append(_histogram([float(x) for x in depths]))
    lines.append("```")
    lines.append("")
    lines.append("### Text length (kept, chars)")
    lines.append("```")
    lines.append(_histogram([float(x) for x in lengths]))
    lines.append("```")
    lines.append("")
    lines.append("### Thread upvotes")
    lines.append("```")
    lines.append(_histogram([float(x) for x in points_list]))
    lines.append("```")
    lines.append("")
    lines.append("## Commentary")
    lines.append("")
    lines.append(commentary)
    lines.append("")

    text = "\n".join(lines)
    out_path.write_text(text, encoding="utf-8")
    return text
