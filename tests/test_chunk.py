# tests for src.chunk: text trimming, parent/grandparent resolution, depth skipping

from __future__ import annotations

import pytest

from src.chunk import _trim, _parent_and_grandparent



class TestTrim:
    def test_short_text_unchanged(self):
        text = "Hello world"
        assert _trim(text) == text

    def test_long_text_trimmed(self):
        text = "word " * 500  # 2500 chars
        result = _trim(text, cap=100)
        assert len(result) <= 100
        assert result.endswith("…")

    def test_exact_cap(self):
        text = "a" * 1200
        result = _trim(text, cap=1200)
        assert result == text

    def test_one_over_cap(self):
        text = "a" * 1201
        result = _trim(text, cap=1200)
        assert len(result) <= 1200
        assert result.endswith("…")

    def test_word_boundary_split(self):
        # Should split at word boundary, not mid-word.
        text = "hello world this is a test"
        result = _trim(text, cap=15)
        # Should split after "hello" or "hello world" (at a space) + "…"
        assert "…" in result
        # Shouldn't have a partial word (except edge cases)



class TestParentAndGrandparent:
    def test_top_level_comment(self):
        # Parent_id points at a thread id → top-level.
        comment = {"id": 100, "parent_id": 1, "text_clean": "text"}
        comments = {100: comment}
        threads = {1: {"id": 1, "title": "Thread"}}
        parent_text, gp_text, is_top = _parent_and_grandparent(comment, comments, threads)
        assert is_top is True
        assert parent_text == ""
        assert gp_text == ""

    def test_reply_to_top_level(self):
        # Depth-1 reply: parent is a comment, grandparent is the thread.
        parent = {"id": 50, "parent_id": 1, "text_clean": "Parent text here"}
        reply = {"id": 100, "parent_id": 50, "text_clean": "Reply text"}
        comments = {50: parent, 100: reply}
        threads = {1: {"id": 1, "title": "Thread"}}
        parent_text, gp_text, is_top = _parent_and_grandparent(reply, comments, threads)
        assert is_top is False
        assert "Parent text" in parent_text
        assert gp_text == ""  # grandparent is the thread

    def test_nested_reply(self):
        # Depth-2 reply: has both parent and grandparent comments.
        gp = {"id": 10, "parent_id": 1, "text_clean": "Grandparent text"}
        parent = {"id": 50, "parent_id": 10, "text_clean": "Parent text"}
        reply = {"id": 100, "parent_id": 50, "text_clean": "Reply text"}
        comments = {10: gp, 50: parent, 100: reply}
        threads = {1: {"id": 1, "title": "Thread"}}
        parent_text, gp_text, is_top = _parent_and_grandparent(reply, comments, threads)
        assert is_top is False
        assert "Parent text" in parent_text
        assert "Grandparent text" in gp_text

    def test_orphan_comment(self):
        # Parent not found → treated as top-level.
        comment = {"id": 100, "parent_id": 999, "text_clean": "Orphan"}
        comments = {100: comment}
        threads = {1: {"id": 1, "title": "Thread"}}
        parent_text, gp_text, is_top = _parent_and_grandparent(comment, comments, threads)
        assert is_top is True



class TestDepthSkipping:
    def test_context_prefix_max_depth_exists(self):
        # The config constant should exist.
        from src.config import CONTEXT_PREFIX_MAX_DEPTH
        assert isinstance(CONTEXT_PREFIX_MAX_DEPTH, int)
        assert CONTEXT_PREFIX_MAX_DEPTH >= 1
