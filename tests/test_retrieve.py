# tests for src.retrieve: query sanitizer, BM25, dense, RRF, re-scorer.
#
# we avoid LLM calls by using only the deterministic pieces (FTS, embeddings,
# fusion, re-scoring). dense path uses a stub embedding model so CI doesn't
# need to download sentence-transformers.

from __future__ import annotations

import math
import sqlite3
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src import retrieve
from tests.conftest import (
    seed_claim,
    seed_comment,
    seed_query,
    seed_thread,
)




class TestFtsQuery:
    def test_simple(self):
        q = retrieve._fts_query("SQLite write performance")
        # tokens should be present (quoted, OR'd)
        assert '"sqlite"' in q
        assert '"write"' in q
        assert '"performance"' in q
        assert " OR " in q

    def test_stopwords_removed(self):
        q = retrieve._fts_query("What is the SQLite story?")
        assert '"sqlite"' in q
        assert '"what"' not in q
        assert '"the"' not in q

    def test_only_stopwords(self):
        assert retrieve._fts_query("the and for what") == '""'

    def test_special_chars_escaped(self):
        # parens / colons should not leak through
        q = retrieve._fts_query("what about (nesting) : WAL mode?")
        assert '"wal"' in q
        assert '"mode"' in q
        assert "(" not in q
        assert ":" not in q

    def test_empty(self):
        assert retrieve._fts_query("") == '""'




class TestRRF:
    def test_single_list(self):
        fused = retrieve.rrf_fuse([(1, 0.1), (2, 0.2)], [], k=60)
        assert fused[0][0] == 1  # top rank gets largest 1/(k+r+1)
        assert fused[1][0] == 2

    def test_merge_boost(self):
        # An id appearing in both lists should outrank ids that appear in one.
        bm = [(1, 0.1), (2, 0.2), (3, 0.3)]
        dn = [(3, 0.9), (1, 0.8)]
        fused = retrieve.rrf_fuse(bm, dn, k=60)
        ids = [c for c, _ in fused]
        assert ids[0] in (1, 3)  # both appear in both lists; top two
        assert 2 in ids          # solo hit still included

    def test_limit_cap(self):
        bm = [(i, 0.01 * i) for i in range(100)]
        fused = retrieve.rrf_fuse(bm, [], limit=5)
        assert len(fused) == 5




class TestRescore:
    def test_features_boost(self, tmp_db):
        qid = seed_query(tmp_db)
        seed_thread(tmp_db, qid, thread_id=100, points=500)
        seed_thread(tmp_db, qid, thread_id=101, points=10)

        seed_comment(tmp_db, comment_id=1, thread_id=100, text_clean="x" * 60)
        seed_comment(tmp_db, comment_id=2, thread_id=101, text_clean="y" * 60)

        # Give descendant_count + firsthand to comment 1 via a firsthand claim.
        conn = sqlite3.connect(str(tmp_db))
        conn.execute("UPDATE comments SET descendant_count = 20 WHERE id = 1")
        conn.commit()
        conn.close()
        seed_claim(tmp_db, comment_id=1, is_firsthand=1)

        # equal rrf_score → comment 1 should win thanks to boosts
        fused = [(1, 0.01), (2, 0.01)]
        out = retrieve.rescore(fused, qid, top_n=2)
        assert out[0]["cid"] == 1
        assert out[0]["final"] > out[1]["final"]

    def test_empty_returns_empty(self, tmp_db):
        assert retrieve.rescore([], 1) == []




class TestBM25:
    def test_returns_hits(self, tmp_db):
        qid = seed_query(tmp_db)
        seed_thread(tmp_db, qid, thread_id=100, title="WAL mode discussion")
        seed_comment(
            tmp_db,
            comment_id=1,
            thread_id=100,
            text_clean="WAL mode handles concurrent readers very well in SQLite.",
            context_prefix="reply about WAL mode",
        )
        seed_comment(
            tmp_db,
            comment_id=2,
            thread_id=100,
            text_clean="Totally unrelated comment about cats.",
            context_prefix="cats",
        )
        # rebuild fts
        conn = sqlite3.connect(str(tmp_db))
        conn.execute("INSERT INTO comments_fts(comments_fts) VALUES('rebuild')")
        conn.commit()
        conn.close()

        hits = retrieve.bm25_search("WAL mode concurrent", qid)
        ids = [cid for cid, _ in hits]
        assert 1 in ids
        # comment 2 may or may not appear, we just require the WAL hit.

    def test_bad_query_returns_empty(self, tmp_db):
        qid = seed_query(tmp_db)
        # empty corpus, no FTS errors and empty result
        assert retrieve.bm25_search("anything", qid) == []




class _FakeEmbed:
    # Deterministic toy embedder: vector = bag-of-characters over a fixed alphabet.

    def encode(self, texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False):
        vecs = []
        for t in texts:
            v = np.zeros(384, dtype=np.float32)
            for i, ch in enumerate((t or "").lower()):
                v[i % 384] += ord(ch) % 17
            n = np.linalg.norm(v) or 1.0
            vecs.append(v / n)
        return np.stack(vecs, axis=0)


class TestDense:
    def test_returns_most_similar(self, tmp_db):
        qid = seed_query(tmp_db)
        seed_thread(tmp_db, qid, thread_id=100)

        seed_comment(tmp_db, comment_id=1, thread_id=100, text_clean="write performance benchmarks")
        seed_comment(tmp_db, comment_id=2, thread_id=100, text_clean="completely different animal facts")

        model = _FakeEmbed()
        vecs = model.encode(
            ["write performance benchmarks", "completely different animal facts"]
        )

        conn = sqlite3.connect(str(tmp_db))
        for cid, row_i in [(1, 0), (2, 1)]:
            conn.execute(
                "INSERT INTO embeddings (comment_id, model, vector) VALUES (?,?,?)",
                (cid, "fake", vecs[row_i].tobytes()),
            )
        conn.commit()
        conn.close()

        with patch.object(retrieve, "_embed", return_value=model):
            hits = retrieve.dense_search("write perf", qid)
        assert hits
        assert hits[0][0] == 1  # "write performance benchmarks" should top the list
