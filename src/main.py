import asyncio
import json
import logging
from pathlib import Path

import typer
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from src import chat as chat_mod
from src import chunk as chunk_mod
from src import cluster as cluster_mod
from src import db
from src import digest as digest_mod
from src import extract as extract_mod
from src import fetch as fetch_mod

logger = logging.getLogger(__name__)

app = FastAPI(title="HN Intel API")


@app.get("/api/generate")
async def generate_digest(topic: str):
    q: asyncio.Queue = asyncio.Queue()

    async def pipeline_worker():
        try:
            print(f"\n[PIPELINE] Started for topic: '{topic}'")
            await q.put(f"data: {json.dumps({'type': 'status', 'message': 'Fetching threads for topic...'})}\n\n")
            summary = await fetch_mod.run_fetch(topic)
            query_id = summary["query_id"]

            await asyncio.to_thread(fetch_mod.write_audit, query_id)

            await q.put(f"data: {json.dumps({'type': 'status', 'message': 'Generating context prefixes...'})}\n\n")
            await chunk_mod.generate_prefixes(query_id, sanity_first=0)

            await q.put(f"data: {json.dumps({'type': 'status', 'message': 'Embedding comments & populating FTS...'})}\n\n")
            await asyncio.to_thread(chunk_mod.embed_comments, query_id)
            await asyncio.to_thread(chunk_mod.populate_fts, query_id)

            await q.put(f"data: {json.dumps({'type': 'status', 'message': 'Extracting structured claims...'})}\n\n")
            await extract_mod.extract_all(query_id)

            await q.put(f"data: {json.dumps({'type': 'status', 'message': 'Clustering claims by stance...'})}\n\n")
            await cluster_mod.cluster_and_label_async(query_id)

            await q.put(f"data: {json.dumps({'type': 'status', 'message': 'Synthesizing final digest...'})}\n\n")
            digest_md = await digest_mod.synthesize(query_id)

            out_path = Path("data") / f"digest_q{query_id}.md"
            out_path.parent.mkdir(exist_ok=True)
            out_path.write_text(digest_md, encoding="utf-8")

            print(f"[PIPELINE] Finished for topic: '{topic}'")
            result = {
                "type": "complete",
                "content": digest_md,
                "query_id": query_id,
                "topic": topic,
            }
            await q.put(f"data: {json.dumps(result)}\n\n")

        except Exception as e:  # noqa: BLE001
            print(f"[PIPELINE] Error: {e}")
            logger.exception("Pipeline error")
            await q.put(f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n")
        finally:
            await q.put(None)

    async def sse_generator():
        task = asyncio.create_task(pipeline_worker())
        while True:
            try:
                msg = await asyncio.wait_for(q.get(), timeout=15)
                if msg is None:
                    break
                yield msg
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"
        await task

    return StreamingResponse(sse_generator(), media_type="text/event-stream")


@app.get("/api/digests")
def list_digests():
    # return recent queries that have a digest on disk
    try:
        with db.connect() as conn:
            rows = conn.execute(
                "SELECT id, topic, fetched_at FROM queries ORDER BY fetched_at DESC LIMIT 20"
            ).fetchall()
    except Exception as e:  # noqa: BLE001
        logger.exception("Failed listing digests")
        raise HTTPException(500, detail=str(e))
    out = []
    for r in rows:
        p = Path("data") / f"digest_q{r['id']}.md"
        if p.exists():
            out.append(
                {
                    "query_id": r["id"],
                    "topic": r["topic"],
                    "fetched_at": r["fetched_at"],
                }
            )
    return {"items": out}


@app.get("/api/digest/{query_id}")
def get_digest(query_id: int):
    p = Path("data") / f"digest_q{query_id}.md"
    if not p.exists():
        raise HTTPException(404, detail=f"no digest for query {query_id}")
    try:
        with db.connect() as conn:
            row = conn.execute(
                "SELECT topic FROM queries WHERE id = ?", (query_id,)
            ).fetchone()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, detail=str(e))
    topic = row["topic"] if row else ""
    return {"query_id": query_id, "topic": topic, "content": p.read_text(encoding="utf-8")}


class ChatStartBody(BaseModel):
    query_id: int


class ChatMessageBody(BaseModel):
    session_id: str
    message: str


@app.post("/api/chat/start")
def chat_start(body: ChatStartBody):
    try:
        sess = chat_mod.start_session(body.query_id)
    except FileNotFoundError as e:
        raise HTTPException(409, detail=str(e))
    except ValueError as e:
        raise HTTPException(404, detail=str(e))
    return {
        "session_id": sess.session_id,
        "topic": sess.topic,
        "query_id": sess.query_id,
    }


@app.post("/api/chat/message")
async def chat_message(body: ChatMessageBody):
    sess = chat_mod.get_session(body.session_id)
    if sess is None:
        raise HTTPException(404, detail="unknown session_id; start a new session")
    msg = (body.message or "").strip()
    if not msg:
        raise HTTPException(400, detail="empty message")
    result = await chat_mod.answer(sess, msg)
    return JSONResponse(result)


frontend_dir = Path(__file__).parent.parent / "frontend"
app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="static")


cli_app = typer.Typer()


@cli_app.command()
def serve(host: str = "127.0.0.1", port: int = 8000):
    # start the web server
    uvicorn.run("src.main:app", host=host, port=port, reload=True)


@cli_app.command()
def run(topic: str):
    # run the full pipeline from CLI

    async def run_pipeline():
        print(f"Running pipeline for '{topic}'...")
        summary = await fetch_mod.run_fetch(topic)
        query_id = summary["query_id"]
        print(f"Fetch done. Query ID: {query_id}")
        await chunk_mod.generate_prefixes(query_id, sanity_first=0)
        chunk_mod.embed_comments(query_id)
        chunk_mod.populate_fts(query_id)
        print("Chunking done.")
        await extract_mod.extract_all(query_id)
        print("Extraction done.")
        await cluster_mod.cluster_and_label_async(query_id)
        print("Clustering done.")
        digest_md = await digest_mod.synthesize(query_id)
        out_path = Path("data") / f"digest_q{query_id}.md"
        out_path.write_text(digest_md, encoding="utf-8")
        print(f"Done! Output written to {out_path}")

    asyncio.run(run_pipeline())


@cli_app.command()
def chat(query_id: int):
    # terminal chat loop

    async def loop():
        sess = chat_mod.start_session(query_id)
        print(f"Chat session started for topic: {sess.topic!r} (query_id={query_id})")
        print("Type a question (blank to quit).\n")
        while True:
            try:
                user = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not user:
                break
            result = await chat_mod.answer(sess, user)
            print(f"\n[intent={result['intent']}  used_retrieval={result['used_retrieval']}]")
            print(result["answer"])
            if result.get("evidence"):
                print("\n(evidence)")
                for e in result["evidence"][:5]:
                    print(f"  [#{e['cid']}] {e['thread_title']}: {e['snippet']}")
            print()

    asyncio.run(loop())


if __name__ == "__main__":
    cli_app()
