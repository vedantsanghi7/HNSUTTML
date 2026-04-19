import asyncio
import sys
import typer
from pathlib import Path

# Fix python path for imports
root_dir = Path(__file__).parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src import fetch as fetch_mod
from src import chunk as chunk_mod
from src import extract as extract_mod
from src import cluster as cluster_mod
from src import digest as digest_mod
from src import chat as chat_mod
from src import llm

app = typer.Typer()

@app.command()
def demo(topic: str = typer.Option("SQLite in production", "--topic")):
    typer.echo(f"\\n=== Starting Demo for topic: '{topic}' ===\\n")
    
    # Run fetch and audit
    summary = asyncio.run(fetch_mod.run_fetch(topic))
    query_id = summary["query_id"]
    fetch_mod.write_audit(query_id)
    
    typer.echo("[Fetch/Audit Summary]")
    typer.echo(f"  Threads fetched: {summary['threads']}")
    typer.echo(f"  Comments kept: {summary['comments_kept']} / {summary['comments_total']}")
    
    # Run chunk
    typer.echo("\\n[Running Context Chunking...]")
    c = asyncio.run(chunk_mod.generate_prefixes(query_id, sanity_first=0))
    chunk_mod.embed_comments(query_id)
    chunk_mod.populate_fts(query_id)
    typer.echo("  Context chunking completed.")
    
    # Run digest
    typer.echo("\\n[Running Digest Pipeline...]")
    asyncio.run(extract_mod.extract_all(query_id, sanity_first=0))
    cluster_mod.cluster_and_label(query_id)
    digest_text = asyncio.run(digest_mod.synthesize(query_id))
    
    digest_path = Path("data") / f"digest_q{query_id}.md"
    digest_path.parent.mkdir(exist_ok=True)
    digest_path.write_text(digest_text, encoding="utf-8")
    
    # Wait to let user read the digest print
    typer.echo("\\n" + "="*40 + "\\nGENERATED DIGEST\\n" + "="*40)
    typer.echo(digest_text)
    typer.echo("="*40 + "\\n")
    
    # Enter chat loop with predefined questions
    typer.echo("\\n[Entering Chat Loop]")
    questions = [
        "What did they say about write performance?",
        "How does this compare to Postgres?",
        "Go back to what you said about WAL mode earlier - any caveats?",
        "What's the weather today?",
    ]
    
    sess = chat_mod.start_session(query_id)
    
    for q in questions:
        typer.echo(f"\\nUSER: {q}")
        res = asyncio.run(chat_mod.answer(sess, q))
        typer.echo(f"ASSISTANT: {res['answer']}")
        
    typer.echo("\\n" + "="*40 + "\\nCOST SUMMARY\\n" + "="*40)
    typer.echo(llm.session_summary())

if __name__ == "__main__":
    app()
