"""Generic split-and-retry batch helper for LLM calls.

Implements the contract in BATCHING_REFACTOR_PLAN.md §3.4–§3.6:
- Try the full batch; on validation failure, split in halves; recurse.
- Single-item failures call a per-stage fallback and return empty.
- Bounded recursion via BATCH_SPLIT_MAX_DEPTH.
"""

from __future__ import annotations

import logging
from typing import Awaitable, Callable, TypeVar

from src.config import BATCH_SPLIT_MAX_DEPTH

T = TypeVar("T")  # input item type
R = TypeVar("R")  # result type

logger = logging.getLogger(__name__)


class BatchValidationError(Exception):
    """Raised by the runner when the batch result fails ID-set or schema checks."""


async def run_with_split_retry(
    items: list[T],
    *,
    item_id: Callable[[T], int],
    run_batch: Callable[[list[T]], Awaitable[dict[int, R]]],
    on_single_failure: Callable[[T], R | None],
    depth: int = 0,
    label: str = "batch",
) -> dict[int, R]:
    """Run `run_batch(items)`; on any exception, split in halves and recurse.

    - `item_id`: how to extract the id from an item (e.g. lambda c: c["id"]).
    - `run_batch`: async function that returns {id: result} or raises on failure.
    - `on_single_failure`: called when a single-item batch fails; returns
      a fallback result (e.g. deterministic prefix) or None to drop.
    - `depth`: current recursion depth; capped at BATCH_SPLIT_MAX_DEPTH.
    """
    if not items:
        return {}

    if len(items) == 1 or depth >= BATCH_SPLIT_MAX_DEPTH:
        try:
            return await run_batch(items)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[%s] single-item or max-depth batch failed (id=%s): %s",
                label, item_id(items[0]) if items else "?", exc,
            )
            # Try the on_single_failure fallback for each item.
            out: dict[int, R] = {}
            for it in items:
                fb = on_single_failure(it)
                if fb is not None:
                    out[item_id(it)] = fb
            return out

    try:
        return await run_batch(items)
    except Exception as exc:  # noqa: BLE001
        logger.info("[%s] batch of %d failed; splitting. err=%s", label, len(items), exc)
        mid = len(items) // 2
        left = await run_with_split_retry(
            items[:mid],
            item_id=item_id,
            run_batch=run_batch,
            on_single_failure=on_single_failure,
            depth=depth + 1,
            label=label,
        )
        right = await run_with_split_retry(
            items[mid:],
            item_id=item_id,
            run_batch=run_batch,
            on_single_failure=on_single_failure,
            depth=depth + 1,
            label=label,
        )
        return {**left, **right}
