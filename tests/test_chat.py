# tests for src.chat: session state, heuristic router, memory assembly,
# evidence formatting, off-topic short-circuit.
# we stub the LLM entirely so no network calls fire.

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from unittest.mock import patch, AsyncMock

import pytest

from src import chat as chat_mod
from src import retrieve
from tests.conftest import seed_query




def _seed_digest(tmp_path: Path, query_id: int, text: str = "# Digest\n\nSample.") -> None:
    p = tmp_path / f"digest_q{query_id}.md"
    p.write_text(text, encoding="utf-8")


@pytest.fixture()
def session(tmp_db, tmp_path, monkeypatch):
    # Build a minimal session pointing at a temp digest file.
    # Redirect DATA_DIR so start_session finds the digest.
    monkeypatch.setattr(chat_mod, "DATA_DIR", tmp_path)

    qid = seed_query(tmp_db, topic="SQLite in production")
    _seed_digest(tmp_path, qid, text="# HN Digest: SQLite in production\n\n## TL;DR\n- [#1] fast")

    sess = chat_mod.start_session(qid)
    return sess




class TestSessionLifecycle:
    def test_start_loads_digest_and_topic(self, session):
        assert session.topic == "SQLite in production"
        assert "HN Digest" in session.digest
        assert session.turns == []
        assert session.rolling_summary == ""

    def test_get_session(self, session):
        got = chat_mod.get_session(session.session_id)
        assert got is session

    def test_missing_digest_raises(self, tmp_db, tmp_path, monkeypatch):
        monkeypatch.setattr(chat_mod, "DATA_DIR", tmp_path)
        qid = seed_query(tmp_db, topic="missing")
        # no digest file on disk
        with pytest.raises(FileNotFoundError):
            chat_mod.start_session(qid)




class TestHeuristicRouter:
    def test_anaphora_triggers_follow_up(self, session):
        d = chat_mod._heuristic_route("What did you say earlier about WAL mode?", session)
        assert d.intent == "follow_up_reference"
        assert d.references_earlier_turn is True
        assert d.requires_retrieval is False

    def test_plain_question_goes_to_howto(self, session):
        d = chat_mod._heuristic_route("Is SQLite reliable for writes?", session)
        assert d.references_earlier_turn is False
        assert d.requires_retrieval is True




class TestEvidenceBlock:
    def test_wraps_in_hn_comment_tags(self):
        rows = [
            {"cid": 42, "text_clean": "lots of text", "thread_title": "SQLite thread",
             "thread_points": 120, "desc_n": 3},
        ]
        block = chat_mod._evidence_block(rows)
        assert '<hn_comment id="42"' in block
        assert "SQLite thread" in block
        assert 'points="120"' in block

    def test_empty(self):
        assert chat_mod._evidence_block([]) == "(no retrieved evidence)"

    def test_trims_long_text(self):
        long_text = "word " * 2000
        rows = [{"cid": 1, "text_clean": long_text, "thread_title": "t",
                 "thread_points": 1, "desc_n": 0}]
        block = chat_mod._evidence_block(rows)
        assert len(block) < len(long_text) + 200  # got trimmed




class TestMemory:
    def test_pinned_digest_first(self, session):
        out = chat_mod._assemble_memory(session)
        assert "Pinned digest" in out
        assert "HN Digest" in out

    def test_recent_turns_appear(self, session):
        session.turns.append(chat_mod.Turn(user="hi", assistant="hello"))
        out = chat_mod._assemble_memory(session)
        assert "USER: hi" in out
        assert "ASSISTANT: hello" in out

    def test_rolling_summary_appears(self, session):
        session.rolling_summary = "summary text"
        out = chat_mod._assemble_memory(session)
        assert "Rolling summary" in out
        assert "summary text" in out

    def test_older_pairs_partition(self, session):
        # 6 turns → 4 recent, 2 older
        for i in range(6):
            session.turns.append(chat_mod.Turn(user=f"u{i}", assistant=f"a{i}"))
        older = session.older_pairs(keep=4)
        recent = session.recent_pairs(4)
        assert len(older) == 2
        assert len(recent) == 4
        assert older[0].user == "u0"
        assert recent[-1].user == "u5"




class TestExtractCiteIds:
    def test_single(self):
        assert chat_mod._extract_cite_ids("see [#123]") == [123]

    def test_multi_in_one(self):
        assert chat_mod._extract_cite_ids("so [#1,#2, 3] end") == [1, 2, 3]

    def test_ignores_junk(self):
        assert chat_mod._extract_cite_ids("no cites here") == []




class TestOffTopic:
    def test_answer_returns_refusal_without_retrieving(self, session, monkeypatch):
        # Router says off_topic; answer() should never hit retrieve or LLM.
        fake_decision = chat_mod.RouterDecision(
            intent="off_topic",
            rewritten_query="weather today",
            requires_retrieval=False,
            references_earlier_turn=False,
        )

        async def fake_route(s, msg):
            return fake_decision

        called = {"retrieve": 0, "llm": 0}

        def fake_hybrid(*a, **kw):
            called["retrieve"] += 1
            return []

        async def fake_llm(*a, **kw):
            called["llm"] += 1
            return {"choices": [{"message": {"content": "should not be called"}}]}

        monkeypatch.setattr(chat_mod, "route", fake_route)
        monkeypatch.setattr(retrieve, "hybrid_retrieve", fake_hybrid)
        monkeypatch.setattr(chat_mod, "llm_chat", fake_llm)

        result = asyncio.run(chat_mod.answer(session, "What's the weather today?"))
        assert result["intent"] == "off_topic"
        assert result["used_retrieval"] is False
        assert called["retrieve"] == 0
        assert called["llm"] == 0
        assert session.topic in result["answer"]
        # turn was still recorded
        assert len(session.turns) == 1
