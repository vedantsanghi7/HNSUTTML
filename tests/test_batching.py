import pytest
from src.batching import run_with_split_retry, BatchValidationError


@pytest.mark.anyio
async def test_full_batch_success():
    async def runner(items):
        return {item["id"]: item["id"] * 10 for item in items}
    out = await run_with_split_retry(
        [{"id": 1}, {"id": 2}, {"id": 3}],
        item_id=lambda x: x["id"],
        run_batch=runner,
        on_single_failure=lambda x: -1,
    )
    assert out == {1: 10, 2: 20, 3: 30}


@pytest.mark.anyio
async def test_one_poison_item_split_retry():
    """If id=2 always fails but others succeed, split-retry should isolate it
    and process the rest."""
    async def runner(items):
        if any(it["id"] == 2 for it in items) and len(items) > 1:
            raise BatchValidationError("poisoned")
        if any(it["id"] == 2 for it in items):
            raise BatchValidationError("still poisoned alone")
        return {it["id"]: it["id"] * 10 for it in items}

    out = await run_with_split_retry(
        [{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}],
        item_id=lambda x: x["id"],
        run_batch=runner,
        on_single_failure=lambda x: -999,
    )
    # 1, 3, 4 should succeed; 2 should hit fallback.
    assert out[1] == 10
    assert out[3] == 30
    assert out[4] == 40
    assert out[2] == -999


@pytest.mark.anyio
async def test_recursion_depth_capped():
    """If every batch fails, we must not recurse infinitely."""
    n_calls = 0
    async def runner(items):
        nonlocal n_calls
        n_calls += 1
        raise BatchValidationError("always")

    out = await run_with_split_retry(
        [{"id": i} for i in range(20)],
        item_id=lambda x: x["id"],
        run_batch=runner,
        on_single_failure=lambda x: -1,
    )
    # All items should fall back; n_calls should be bounded (< 100).
    assert all(v == -1 for v in out.values())
    assert n_calls < 100


@pytest.mark.anyio
async def test_empty_batch_returns_empty():
    """Empty input should return empty dict without calling the runner."""
    called = False
    async def runner(items):
        nonlocal called
        called = True
        return {}

    out = await run_with_split_retry(
        [],
        item_id=lambda x: x["id"],
        run_batch=runner,
        on_single_failure=lambda x: -1,
    )
    assert out == {}
    assert not called


@pytest.mark.anyio
async def test_fallback_none_drops_item():
    """If on_single_failure returns None, that item should be dropped."""
    async def runner(items):
        raise BatchValidationError("fail")

    out = await run_with_split_retry(
        [{"id": 1}],
        item_id=lambda x: x["id"],
        run_batch=runner,
        on_single_failure=lambda x: None,
    )
    assert out == {}
