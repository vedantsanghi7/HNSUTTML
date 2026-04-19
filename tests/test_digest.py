"""Tests for src.digest: topic tokens, off-topic filter, citations."""

from __future__ import annotations

import pytest

from src.digest import (
    _all_bullets_have_cite,
    _compile_alternatives,
    _extract_cites,
    _filter_off_topic,
    _firsthand_war_stories,
    _score_cluster_relevance,
    _topic_tokens,
)



class TestTopicTokens:
    def test_basic(self):
        tokens = _topic_tokens("SQLite in production")
        assert "sqlite" in tokens
        assert "production" in tokens

    def test_stop_words_removed(self):
        tokens = _topic_tokens("the and for with that this")
        assert tokens == []

    def test_short_words_excluded(self):
        # Words < 3 chars (after regex) are not captured.
        tokens = _topic_tokens("AI in ML")
        # 'AI' and 'ML' are only 2 chars → excluded by the regex [A-Za-z][A-Za-z0-9+]{2,}
        assert "ai" not in tokens
        assert "ml" not in tokens

    def test_compound_word_splitting(self):
        # CamelCase / mixed-case should produce sub-tokens.
        tokens = _topic_tokens("macOS")
        assert "macos" in tokens
        assert "mac" in tokens

    def test_gemini_macos(self):
        # The specific failing query.
        tokens = _topic_tokens("Gemini for MACos")
        assert "gemini" in tokens
        assert "macos" in tokens

    def test_plus_in_token(self):
        tokens = _topic_tokens("C++ performance")
        assert "c++" in tokens
        assert "performance" in tokens

    def test_empty(self):
        assert _topic_tokens("") == []

    def test_all_stop_words(self):
        assert _topic_tokens("the and for") == []



class TestScoreClusterRelevance:
    def test_all_match(self):
        cluster = {"members": [
            {"thread_id": 1},
            {"thread_id": 1},
        ]}
        titles = {1: "SQLite in production"}
        assert _score_cluster_relevance(cluster, titles, ["sqlite"]) == 1.0

    def test_none_match(self):
        cluster = {"members": [
            {"thread_id": 1},
            {"thread_id": 2},
        ]}
        titles = {1: "Unrelated thread", 2: "Another unrelated"}
        assert _score_cluster_relevance(cluster, titles, ["sqlite"]) == 0.0

    def test_partial_match(self):
        cluster = {"members": [
            {"thread_id": 1},
            {"thread_id": 2},
        ]}
        titles = {1: "SQLite rocks", 2: "Unrelated thread"}
        assert _score_cluster_relevance(cluster, titles, ["sqlite"]) == 0.5

    def test_empty_members(self):
        cluster = {"members": []}
        assert _score_cluster_relevance(cluster, {}, ["sqlite"]) == 1.0



class TestFilterOffTopic:
    def _cluster(self, thread_id: int) -> dict:
        return {
            "members": [
                {"thread_id": thread_id, "comment_id": 100, "claim_text": "test",
                 "confidence": 0.8, "stance": "pro", "is_firsthand": False,
                 "tools_mentioned": "[]"},
            ]
        }

    def test_all_on_topic(self):
        clusters = [self._cluster(1), self._cluster(1)]
        titles = {1: "SQLite in production"}
        result = _filter_off_topic(clusters, titles, ["sqlite"])
        assert len(result) == 2

    def test_all_off_topic_keeps_all(self):
        # The main bug fix: should NOT crash, should keep all clusters.
        clusters = [self._cluster(1), self._cluster(2)]
        titles = {1: "Browser Use", 2: "Dayflow"}
        result = _filter_off_topic(clusters, titles, ["gemini", "macos"])
        # should keep all, graceful fallback
        assert len(result) == 2

    def test_partial_strict_threshold(self):
        # Clusters above 50% relevance are kept, others dropped.
        on_topic = {"members": [
            {"thread_id": 1}, {"thread_id": 1}, {"thread_id": 2}
        ]}
        off_topic = {"members": [
            {"thread_id": 2}, {"thread_id": 2}, {"thread_id": 2}
        ]}
        titles = {1: "SQLite discussion", 2: "Unrelated"}
        result = _filter_off_topic([on_topic, off_topic], titles, ["sqlite"])
        assert len(result) == 1
        assert result[0] is on_topic

    def test_empty_tokens_returns_all(self):
        clusters = [self._cluster(1)]
        result = _filter_off_topic(clusters, {}, [])
        assert len(result) == 1

    def test_empty_clusters_returns_empty(self):
        result = _filter_off_topic([], {1: "test"}, ["sqlite"])
        assert result == []

    def test_relaxed_threshold_fallback(self):
        # If no cluster passes 50%, try 30%.
        # 1 of 3 members on-topic = 33% → passes 30% but not 50%
        cluster = {"members": [
            {"thread_id": 1}, {"thread_id": 2}, {"thread_id": 2}
        ]}
        titles = {1: "SQLite thread", 2: "Other"}
        result = _filter_off_topic([cluster], titles, ["sqlite"])
        assert len(result) == 1



class TestExtractCites:
    def test_single_cite(self):
        assert _extract_cites("text [#123] more") == [123]

    def test_multiple_cites(self):
        result = _extract_cites("[#1] and [#2] and [#3]")
        assert result == [1, 2, 3]

    def test_comma_separated(self):
        result = _extract_cites("[#1,#2,#3]")
        assert result == [1, 2, 3]

    def test_no_cites(self):
        assert _extract_cites("no citations here") == []

    def test_mixed(self):
        result = _extract_cites("[#100] text [#200,#300]")
        assert 100 in result
        assert 200 in result
        assert 300 in result



class TestAllBulletsHaveCite:
    def test_all_cited(self):
        text = "# Title\n- Point one [#1]\n- Point two [#2]"
        assert _all_bullets_have_cite(text) is True

    def test_missing_cite(self):
        text = "- Point with cite [#1]\n- Point without cite"
        assert _all_bullets_have_cite(text) is False

    def test_no_bullets(self):
        text = "Just a paragraph with no bullets"
        assert _all_bullets_have_cite(text) is True

    def test_empty(self):
        assert _all_bullets_have_cite("") is True



class TestCompileAlternatives:
    def test_empty(self):
        assert _compile_alternatives([]) == []

    def test_counts_tools(self):
        clusters = [{
            "members": [
                {"tools_mentioned": '["Redis", "Postgres"]', "comment_id": 1, "claim_text": "claim"},
                {"tools_mentioned": '["Redis"]', "comment_id": 2, "claim_text": "claim2"},
            ]
        }]
        result = _compile_alternatives(clusters)
        redis = next(r for r in result if r["display"] == "Redis")
        assert redis["count"] == 2
        postgres = next(r for r in result if r["display"] == "Postgres")
        assert postgres["count"] == 1

    def test_skips_self_tools(self):
        # Tools in _SELF_TOOL_TOKENS should be excluded.
        clusters = [{
            "members": [
                {"tools_mentioned": '["SQLite", "Redis"]', "comment_id": 1, "claim_text": "claim"},
            ]
        }]
        result = _compile_alternatives(clusters)
        tool_names = [r["display"].lower() for r in result]
        assert "sqlite" not in tool_names
        assert "redis" in [r["display"].lower() for r in result]

    def test_malformed_json_skipped(self):
        clusters = [{
            "members": [
                {"tools_mentioned": "not json", "comment_id": 1, "claim_text": "c"},
            ]
        }]
        # Should not crash
        assert _compile_alternatives(clusters) == []



class TestFirsthandWarStories:
    def _member(self, cid: int, firsthand: bool, stance: str, conf: float) -> dict:
        return {
            "comment_id": cid,
            "claim_text": f"Story from {cid}",
            "is_firsthand": firsthand,
            "stance": stance,
            "confidence": conf,
            "evidence_type": "anecdote",
        }

    def test_picks_firsthand(self):
        clusters = [{"members": [
            self._member(1, True, "anecdote", 0.9),
            self._member(2, False, "anecdote", 0.95),  # not firsthand
            self._member(3, True, "pro", 0.7),
        ]}]
        result = _firsthand_war_stories(clusters)
        cids = [r["comment_id"] for r in result]
        assert 1 in cids
        assert 3 in cids
        assert 2 not in cids

    def test_respects_limit(self):
        clusters = [{"members": [
            self._member(i, True, "anecdote", 0.5) for i in range(20)
        ]}]
        result = _firsthand_war_stories(clusters, limit=3)
        assert len(result) == 3

    def test_dedupes_by_comment_id(self):
        clusters = [
            {"members": [self._member(1, True, "anecdote", 0.9)]},
            {"members": [self._member(1, True, "anecdote", 0.8)]},
        ]
        result = _firsthand_war_stories(clusters)
        assert len(result) == 1

    def test_empty(self):
        assert _firsthand_war_stories([]) == []
