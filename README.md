# HN Thread Intelligence Tool

Give it a topic, it fetches HN threads, extracts structured claims, clusters them by stance, and produces a cited digest. Then you can ask follow-ups grounded in what the community actually said.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your SARVAM_API_KEY

# run the full pipeline
python -m src.main run "SQLite in production"

# or start the web UI
python -m src.main serve
```

Demo script (runs pipeline + 4 test chat questions):
```bash
python scripts/run_demo.py --topic "SQLite in production"
```

## Pipeline

```
fetch -> chunk -> extract -> cluster -> digest -> chat
```

1. **Fetch** - Multi-axis Algolia search (relevance, points, recency, Ask HN) to avoid echo chambers. Full comment trees via Firebase.
2. **Chunk** - Context prefix per comment + bge-small embeddings + FTS5 index.
3. **Extract** - Sarvam-30B tool calling -> structured claims (stance, evidence type, confidence, tools mentioned). Pydantic validated.
4. **Cluster** - HDBSCAN on claim embeddings, weighted by community signal.
5. **Digest** - Sarvam-105B synthesis with citation verification.
6. **Chat** - Hybrid retrieval (BM25 + dense via RRF) with layered memory and groundedness checks.

## Design decisions

### Data audit

The fetch stage pulls threads across 4 axes (relevance, top-by-points, recent, Ask HN) because searching only by relevance gives you an echo chamber. We normalize HTML, compute descendant counts for signal scoring, and discard short non-code comments while keeping code-bearing ones regardless of length since code snippets are often the most concrete evidence. The audit report (`python -m src.cli audit --query-id <id>`) dumps per-thread stats, depth/length histograms, and commentary on what got filtered and why.

### Chunking strategy

HN comments aren't documents, they're nodes in a conversation graph. A reply like "I agree, but only for small datasets" is meaningless without its parent. So instead of token-splitting, our unit is the comment, and each one gets a one-sentence context prefix describing what it's replying to. Top-level comments get a deterministic prefix. Replies get an LLM-generated one from Sarvam-30B using the parent + grandparent text. We embed `prefix + comment` with bge-small-en-v1.5 so the vector captures conversational position, not just content.

Optimizations: deep comments (depth > 2) and long comments (>250 chars) skip the LLM prefix since they're either too far from the root to benefit or dense enough to stand alone. This cuts API calls by ~40% without hurting retrieval quality.

### Context management

Chat uses layered memory: the digest is always pinned (tier 1), a rolling summary compresses older turns every 4 exchanges (tier 2), and the last 4 turns are kept verbatim (tier 3). This keeps the context window bounded while preserving references for follow-ups.

The query router (Sarvam-30B tool call) classifies intent and rewrites anaphoric queries before retrieval. If the user says "what you mentioned earlier," the router flags `references_earlier_turn=true` and skips retrieval in favor of the memory tiers.

### Edge cases

- **No answer in data** - Off-topic questions get a deterministic refusal without hitting retrieval or the answer LLM. Router classifies `off_topic` and short-circuits.
- **Contradictory opinions** - The cluster stage groups opposing stances (pro vs con in the same category). The digest template has a dedicated "Debates" section that surfaces these, citing both sides.
- **Prompt injection** - Untrusted comment text lives inside `<hn_comment>` XML tags with explicit system instructions to ignore anything inside them.
- **Groundedness** - A verifier checks each cited sentence against its source comment. If any claim is unsupported, the answer regenerates with reinforcement.

### Honest limitations

- Multi-axis fan-out sometimes pulls tangential threads. We filter at cluster-time but the fetch stage could be smarter.
- HDBSCAN leaves ~30% of claims as noise (singletons). They're excluded from the digest but still surface in chat retrieval.
- The 3-feature re-scorer is hand-tuned, not learned. A cross-encoder would be better.
- Groundedness verifier uses the same provider as the answerer, so there's some correlation bias.

## Tests

```bash
pytest tests/ -v   # 117 tests
```
