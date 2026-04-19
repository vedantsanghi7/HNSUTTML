# async sarvam client with caching, retry, call logging, and cost summary.
#
# both sarvam-30b and 105b are reasoning models, so they burn tokens on a
# hidden reasoning preamble before the actual content. keep max_tokens
# generous (>=1500 for tool calls).

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import traceback
from pathlib import Path
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from src import db
from src.config import (
    CACHE_DIR,
    LLM_CONCURRENCY,
    LLM_MAX_RETRIES,
    PROMPT_VERSION,
    SARVAM_AUTH_HEADER,
    SARVAM_BASE_URL,
    settings,
)
from src.fetch import log_error

# semaphore is tied to its event loop. since the CLI can run multiple
# asyncio.run() calls (one per stage), we key by loop id to avoid issues.
_sem_by_loop: dict[int, asyncio.Semaphore] = {}


def _semaphore() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    key = id(loop)
    sem = _sem_by_loop.get(key)
    if sem is None:
        sem = asyncio.Semaphore(LLM_CONCURRENCY)
        _sem_by_loop[key] = sem
    return sem


# sarvam starter tier caps max_tokens at 4096
MAX_TOKENS_CEILING = 4096


def _cache_key(
    *,
    model: str,
    messages: list[dict],
    tools: list[dict] | None,
    tool_choice: Any,
    temperature: float,
) -> str:
    blob = json.dumps(
        {
            "v": PROMPT_VERSION,
            "model": model,
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice,
            "temperature": temperature,
        },
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _cache_path(key: str) -> Path:
    p = CACHE_DIR / f"{key}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return code == 429 or 500 <= code < 600
    return isinstance(exc, (httpx.RequestError, httpx.TimeoutException))


@retry(
    retry=retry_if_exception(_is_transient),
    wait=wait_exponential_jitter(initial=2, max=30),
    stop=stop_after_attempt(LLM_MAX_RETRIES),
    reraise=True,
)
async def _post(
    client: httpx.AsyncClient, payload: dict, timeout: float
) -> dict:
    r = await client.post(
        f"{SARVAM_BASE_URL}/chat/completions",
        headers={
            SARVAM_AUTH_HEADER: settings.sarvam_api_key,
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    # 4xx (except 429) won't be retried, surface the body for debugging.
    if 400 <= r.status_code < 500 and r.status_code != 429:
        raise httpx.HTTPStatusError(
            f"{r.status_code} {r.reason_phrase}: {r.text[:500]}",
            request=r.request,
            response=r,
        )
    r.raise_for_status()
    return r.json()


def _log_call(
    *,
    model: str,
    purpose: str,
    usage: dict | None,
    latency_ms: int,
    cache_hit: bool,
    prompt_sha: str,
) -> None:
    usage = usage or {}
    with db.connect() as conn:
        conn.execute(
            """
            INSERT INTO llm_calls (ts, model, purpose, tokens_in, tokens_out,
                                   latency_ms, cache_hit, prompt_sha)
            VALUES (?,?,?,?,?,?,?,?)
            """,
            (
                int(time.time()),
                model,
                purpose,
                usage.get("prompt_tokens"),
                usage.get("completion_tokens"),
                latency_ms,
                1 if cache_hit else 0,
                prompt_sha,
            ),
        )


async def chat(
    messages: list[dict],
    *,
    purpose: str,
    model: str,
    tools: list[dict] | None = None,
    tool_choice: Any = None,
    temperature: float = 0.3,
    max_tokens: int = 4096,
    client: httpx.AsyncClient | None = None,
) -> dict:
    # Invoke Sarvam chat completions. Returns the full response JSON.
    # Cached on exact (model, messages, tools, tool_choice, temperature, version).
    if not settings.sarvam_api_key:
        raise RuntimeError("SARVAM_API_KEY not set; configure .env")

    key = _cache_key(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        temperature=temperature,
    )
    path = _cache_path(key)

    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            _log_call(
                model=model,
                purpose=purpose,
                usage=data.get("usage"),
                latency_ms=0,
                cache_hit=True,
                prompt_sha=key,
            )
            return data
        except Exception:  # noqa: BLE001 - corrupt cache, just re-fetch
            pass

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": min(max_tokens, MAX_TOKENS_CEILING),
    }
    if tools:
        payload["tools"] = tools
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice

    sem = _semaphore()
    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient()

    t0 = time.time()
    try:
        async with sem:
            data = await _post(client, payload, timeout=90.0)
    except Exception as exc:  # noqa: BLE001
        log_error(
            "llm.chat",
            f"{purpose}: {type(exc).__name__}: {exc}",
            traceback.format_exc(),
            inp=f"model={model} key={key[:12]}",
        )
        raise
    finally:
        if owns_client:
            await client.aclose()

    latency_ms = int((time.time() - t0) * 1000)

    # don't cache truncated responses, caller should retry with more tokens
    finish = (data.get("choices") or [{}])[0].get("finish_reason")
    if finish != "length":
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    _log_call(
        model=model,
        purpose=purpose,
        usage=data.get("usage"),
        latency_ms=latency_ms,
        cache_hit=False,
        prompt_sha=key,
    )
    return data


def extract_text(response: dict) -> str:
    # Best-effort text extraction from a chat completion response.
    choice = response["choices"][0]["message"]
    return (choice.get("content") or "").strip()


def extract_tool_args(response: dict, tool_name: str) -> dict | None:
    # Return parsed JSON arguments for the named tool call, or None.
    choice = response["choices"][0]["message"]
    for call in choice.get("tool_calls") or []:
        fn = call.get("function") or {}
        if fn.get("name") == tool_name:
            try:
                return json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                return None
    return None


def session_summary() -> str:
    # Return a human-readable summary of llm_calls for this DB.
    with db.connect() as conn:
        rows = conn.execute(
            """
            SELECT model, purpose,
                   COUNT(*) AS calls,
                   SUM(cache_hit) AS hits,
                   SUM(COALESCE(tokens_in,0)) AS tin,
                   SUM(COALESCE(tokens_out,0)) AS tout,
                   AVG(latency_ms) AS avg_ms
            FROM llm_calls
            GROUP BY model, purpose
            ORDER BY model, purpose
            """
        ).fetchall()
    if not rows:
        return "(no LLM calls recorded)"
    lines = ["model             purpose              calls  hits   tok_in  tok_out  avg_ms"]
    for r in rows:
        lines.append(
            f"{r['model']:<17} {r['purpose']:<20} {r['calls']:>5} {r['hits'] or 0:>5} "
            f"{r['tin']:>8} {r['tout']:>8} {int(r['avg_ms'] or 0):>7}"
        )
    return "\n".join(lines)
