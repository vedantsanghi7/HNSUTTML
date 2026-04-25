# claim clustering + weighting + canonical labels.
#
# embeds claim_text with bge-small, runs HDBSCAN on the vectors,
# weights clusters by community signal, and gets a canonical label
# from sarvam-30b for each cluster (batched).

from __future__ import annotations

import asyncio
import json
import math
import statistics
from collections import Counter

import httpx
import numpy as np
from pydantic import BaseModel, ValidationError

from src import db
from src.chunk import _embed
from src.config import (
    CLUSTER_LABEL_BATCH_SIZE,
    EMIT_LABELS_BATCH_TOOL,
    MODEL_SMALL,
    P3_CLUSTER_LABEL,
    P3_CLUSTER_LABEL_BATCH,
)
from src.llm import chat, extract_tool_args


# claim embeddings


def embed_claims() -> tuple[list[int], np.ndarray]:
    with db.connect() as conn:
        rows = conn.execute(
            """
            SELECT c.id, c.claim_text FROM claims c
            LEFT JOIN claim_embeddings e ON e.claim_id = c.id
            """
        ).fetchall()
    if not rows:
        return [], np.zeros((0, 384), dtype=np.float32)
    model = _embed()
    vecs = model.encode(
        [r["claim_text"] for r in rows],
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    vecs = np.asarray(vecs, dtype=np.float32)
    with db.connect() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO claim_embeddings (claim_id, vector) VALUES (?,?)",
            [(rows[i]["id"], vecs[i].tobytes()) for i in range(len(rows))],
        )
    return [r["id"] for r in rows], vecs


# clustering


def _run_hdbscan(vecs: np.ndarray) -> np.ndarray:
    import hdbscan

    n = len(vecs)
    # min_cluster_size=3 default; drop to 2 when claim count is small
    min_cluster_size = 2 if n < 80 else 3
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=2,
        metric="euclidean",
    )
    return clusterer.fit_predict(vecs)


# weighting


_STANCE_TO_NUM = {
    "pro": 1.0,
    "con": -1.0,
    "alternative": -0.5,
    "neutral": 0.0,
    "anecdote": 0.0,
    "benchmark": 0.0,
}


def _variance_of_stances(stances: list[str]) -> float:
    nums = [_STANCE_TO_NUM.get(s, 0.0) for s in stances]
    if len(nums) < 2:
        return 0.0
    return statistics.pvariance(nums)


def compute_cluster_payloads(
    query_id: int, claim_ids: list[int], labels: np.ndarray
) -> list[dict]:
    # For each non-noise cluster: compute weight, pick top claims, collect metadata.
    with db.connect() as conn:
        claim_rows = {
            r["id"]: dict(r)
            for r in conn.execute(
                """
                SELECT cl.id, cl.comment_id, cl.claim_text, cl.stance, cl.category,
                       cl.evidence_type, cl.confidence, cl.is_firsthand,
                       c.author, c.thread_id,
                       t.points AS thread_points
                FROM claims cl
                JOIN comments c ON c.id = cl.comment_id
                JOIN threads t ON t.id = c.thread_id
                WHERE t.query_id = ?
                """,
                (query_id,),
            ).fetchall()
        }

    label_by_claim = dict(zip(claim_ids, labels.tolist()))

    clusters: dict[int, list[int]] = {}
    for cid, lab in label_by_claim.items():
        if lab < 0:
            continue
        clusters.setdefault(int(lab), []).append(cid)

    payloads: list[dict] = []
    for lab, members in clusters.items():
        rs = [claim_rows[m] for m in members if m in claim_rows]
        if not rs:
            continue
        authors = {r["author"] for r in rs if r["author"]}
        thread_points = {r["thread_id"]: r["thread_points"] or 0 for r in rs}
        stance = Counter(r["stance"] for r in rs).most_common(1)[0][0]
        category = Counter(r["category"] for r in rs).most_common(1)[0][0]
        firsthand = sum(1 for r in rs if r["is_firsthand"])
        benchmarks = sum(1 for r in rs if r["evidence_type"] == "benchmark")
        var = _variance_of_stances([r["stance"] for r in rs])

        weight = (
            sum(math.log1p(p) for p in thread_points.values()) * 0.5
            + len(authors) * 1.0
            + firsthand * 0.8
            + benchmarks * 0.6
            - var * 1.5
        )
        top3 = sorted(rs, key=lambda r: r["confidence"], reverse=True)[:3]
        payloads.append(
            {
                "member_claim_ids": [r["id"] for r in rs],
                "members": rs,
                "stance": stance,
                "category": category,
                "weight": round(weight, 3),
                "n_authors": len(authors),
                "n_firsthand": firsthand,
                "n_benchmarks": benchmarks,
                "variance": round(var, 3),
                "top3": top3,
            }
        )

    payloads.sort(key=lambda p: p["weight"], reverse=True)
    return payloads


# --- Batched canonical labels ---


class _LabelItem(BaseModel):
    cluster_id: int
    label: str

class _LabelBatch(BaseModel):
    labels: list[_LabelItem]


def _build_label_batch_user_msg(payloads: list[dict], offset: int) -> str:
    """`offset` lets us renumber clusters across batches so cluster_id is unique
    per LLM call. Returns the user message."""
    parts: list[str] = []
    for i, p in enumerate(payloads):
        cid = offset + i
        # Reuse the top-3 highest-confidence claims as before.
        top3 = p["top3"]
        while len(top3) < 3:
            top3 = top3 + [top3[-1]]
        parts.append(
            f'<cluster id="{cid}">\n'
            f'1. {top3[0]["claim_text"]}\n'
            f'2. {top3[1]["claim_text"]}\n'
            f'3. {top3[2]["claim_text"]}\n'
            f'</cluster>'
        )
    return "\n\n".join(parts)


async def _label_all(payloads: list[dict]) -> None:
    """Assign a one-sentence label to each cluster payload. Labels are written
    onto each payload dict in-place under key 'label'."""
    if not payloads:
        return

    async with httpx.AsyncClient() as client:
        for batch_start in range(0, len(payloads), CLUSTER_LABEL_BATCH_SIZE):
            batch = payloads[batch_start : batch_start + CLUSTER_LABEL_BATCH_SIZE]
            local_ids = list(range(batch_start, batch_start + len(batch)))
            user_msg = _build_label_batch_user_msg(batch, offset=batch_start)
            try:
                resp = await chat(
                    [
                        {"role": "system", "content": P3_CLUSTER_LABEL_BATCH},
                        {"role": "user", "content": user_msg},
                    ],
                    purpose="cluster_label_batch",
                    model=MODEL_SMALL,
                    tools=[EMIT_LABELS_BATCH_TOOL],
                    tool_choice={"type": "function", "function": {"name": "emit_labels"}},
                    temperature=0.2,
                    max_tokens=2500,
                    client=client,
                )
                args = extract_tool_args(resp, "emit_labels")
                parsed = _LabelBatch.model_validate(args or {})
                got = {entry.cluster_id: entry.label.strip().strip('"') for entry in parsed.labels}
            except (ValidationError, Exception):  # noqa: BLE001
                got = {}

            for local_id, payload in zip(local_ids, batch):
                lab = got.get(local_id)
                if not lab:
                    lab = payload["top3"][0]["claim_text"][:160]
                payload["label"] = lab


# public entry point


def cluster_and_label(query_id: int) -> dict:
    return asyncio.run(cluster_and_label_async(query_id))

async def cluster_and_label_async(query_id: int) -> dict:
    loop = asyncio.get_running_loop()
    # Run heavy blocking tasks in thread pool
    claim_ids, vecs = await loop.run_in_executor(None, embed_claims)
    if len(claim_ids) == 0:
        return {"clusters": 0, "noise": 0}

    labels = await loop.run_in_executor(None, _run_hdbscan, vecs)
    noise = int((labels < 0).sum())

    payloads = await loop.run_in_executor(None, compute_cluster_payloads, query_id, claim_ids, labels)
    await _label_all(payloads)

    # Persist (wipe prior clusters for this query first)
    def _persist():
        with db.connect() as conn:
            conn.execute("DELETE FROM clusters WHERE query_id = ?", (query_id,))
            conn.executemany(
                """
                INSERT INTO clusters (query_id, label, stance, category, weight, member_claim_ids)
                VALUES (?,?,?,?,?,?)
                """,
                [
                    (
                        query_id,
                        p.get("label") or p["top3"][0]["claim_text"][:160],
                        p["stance"],
                        p["category"],
                        float(p["weight"]),
                        json.dumps(p["member_claim_ids"]),
                    )
                    for p in payloads
                ],
            )
    await loop.run_in_executor(None, _persist)

    return {"clusters": len(payloads), "noise": noise}
