# CLI entry point (typer). commands: fetch, audit, chunk, digest.

from __future__ import annotations

import asyncio

import typer

from src import fetch as fetch_mod

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _emit_cost_summary() -> None:
    from src.llm import session_summary

    typer.echo("\n--- LLM session summary ---")
    typer.echo(session_summary())


@app.command()
def fetch(topic: str = typer.Argument(..., help="Topic to search on HN")) -> None:
    # Fetch threads for TOPIC across the multi-axis fan-out and persist to SQLite.
    summary = asyncio.run(fetch_mod.run_fetch(topic))
    if summary.get("cached"):
        typer.echo(
            f"[cached] query_id={summary['query_id']} threads={summary['threads']} "
            f"kept={summary['comments_kept']}/{summary['comments_total']} "
            f"(re-fetch skipped within idempotency window)"
        )
    else:
        typer.echo(
            f"[fetched] query_id={summary['query_id']} threads={summary['threads']} "
            f"kept={summary['comments_kept']}/{summary['comments_total']} "
            f"slots={dict(summary['picked_slots'])}"
        )


@app.command()
def audit(
    query_id: int = typer.Option(..., "--query-id", help="Query id to audit"),
) -> None:
    # Write data/audit_report.md for QUERY_ID.
    fetch_mod.write_audit(query_id)
    typer.echo(f"audit written: data/audit_report.md (query_id={query_id})")


@app.command()
def chunk(
    query_id: int = typer.Option(..., "--query-id", help="Query id to chunk"),
    sanity_first: int = typer.Option(5, help="Inspect first N prefixes before full batch"),
) -> None:
    # Stage 2: context prefixes + embeddings + FTS.
    import asyncio

    from src import chunk as chunk_mod

    p = asyncio.run(chunk_mod.generate_prefixes(query_id, sanity_first=sanity_first))
    typer.echo(f"[chunk] prefixes: {p['with_prefix']}/{p['active']}")
    e = chunk_mod.embed_comments(query_id)
    typer.echo(f"[chunk] embeddings: {e['embedded']} x {e['dim']}-dim")
    f = chunk_mod.populate_fts(query_id)
    typer.echo(f"[chunk] fts rows: {f['indexed']}")
    _emit_cost_summary()


@app.command()
def digest(
    query_id: int = typer.Option(..., "--query-id", help="Query id to digest"),
    out: str = typer.Option("data/digest.md", help="Output path"),
) -> None:
    # Stage 3: extract claims, cluster, synthesize digest.
    import asyncio
    from pathlib import Path

    from src import cluster as cluster_mod
    from src import digest as digest_mod
    from src import extract as extract_mod

    ex = asyncio.run(extract_mod.extract_all(query_id))
    typer.echo(
        f"[digest] claims: {ex['claims']} from {ex['substantive']}/{ex['scanned']} substantive comments"
    )
    cl = cluster_mod.cluster_and_label(query_id)
    typer.echo(f"[digest] clusters: {cl['clusters']} (noise claims: {cl['noise']})")
    text = asyncio.run(digest_mod.synthesize(query_id))
    Path(out).write_text(text, encoding="utf-8")
    typer.echo(f"[digest] written: {out}")
    _emit_cost_summary()


if __name__ == "__main__":
    app()
