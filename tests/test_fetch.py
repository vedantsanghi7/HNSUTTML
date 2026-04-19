"""Tests for src.fetch: normalization, story filter, descendant counts."""

from __future__ import annotations

import pytest

from src.fetch import (
    CommentNode,
    _compute_descendant_counts,
    _story_filter,
    normalize_comment_html,
)



class TestNormalizeCommentHtml:
    def test_plain_text(self):
        text, has_code, qd = normalize_comment_html("Hello world")
        assert text == "Hello world"
        assert has_code is False

    def test_paragraph_breaks(self):
        text, _, _ = normalize_comment_html("<p>First</p><p>Second</p>")
        assert "First" in text
        assert "Second" in text
        # Should have paragraph breaks (double newline)
        assert "\n\n" in text

    def test_code_preserved(self):
        text, has_code, _ = normalize_comment_html("<pre><code>print('hi')</code></pre>")
        assert has_code is True
        assert "print('hi')" in text
        assert "```" in text  # fenced code

    def test_inline_code(self):
        text, has_code, _ = normalize_comment_html("Use <code>sqlite3</code> for this.")
        assert has_code is True
        assert "`sqlite3`" in text

    def test_entities_decoded(self):
        text, _, _ = normalize_comment_html("&amp; &lt; &gt; &quot;")
        assert "&" in text
        assert "<" in text

    def test_link_preserved(self):
        text, _, _ = normalize_comment_html('<a href="https://example.com">click here</a>')
        assert "click here" in text
        assert "https://example.com" in text

    def test_empty_input(self):
        text, has_code, qd = normalize_comment_html("")
        assert text == ""
        assert has_code is False
        assert qd == 0.0

    def test_quote_density(self):
        # All lines start with >
        text, _, qd = normalize_comment_html("> quote1<p>> quote2")
        assert qd > 0

    def test_mixed_content(self):
        html = "<p>Regular text</p><pre><code>code block</code></pre><p>More text</p>"
        text, has_code, _ = normalize_comment_html(html)
        assert has_code is True
        assert "Regular text" in text
        assert "More text" in text
        assert "code block" in text



class TestStoryFilter:
    def test_valid_story(self):
        hit = {"objectID": "123", "num_comments": 50}
        assert _story_filter(hit) is True

    def test_too_few_comments(self):
        hit = {"objectID": "123", "num_comments": 5}
        assert _story_filter(hit) is False

    def test_no_objectid(self):
        hit = {"num_comments": 100}
        assert _story_filter(hit) is False

    def test_dead_story(self):
        hit = {"objectID": "123", "num_comments": 100, "dead": True}
        assert _story_filter(hit) is False

    def test_deleted_story(self):
        hit = {"objectID": "123", "num_comments": 100, "deleted": True}
        assert _story_filter(hit) is False

    def test_exact_min_comments(self):
        from src.config import MIN_NUM_COMMENTS
        hit = {"objectID": "123", "num_comments": MIN_NUM_COMMENTS}
        assert _story_filter(hit) is True

    def test_below_min_comments(self):
        from src.config import MIN_NUM_COMMENTS
        hit = {"objectID": "123", "num_comments": MIN_NUM_COMMENTS - 1}
        assert _story_filter(hit) is False



class TestComputeDescendantCounts:
    def _node(self, nid: int, pid: int | None = None) -> CommentNode:
        return CommentNode(
            id=nid, parent_id=pid, author="u", created_at=0,
            text_raw="", depth=0, kids=[], dead=False, deleted=False,
        )

    def test_flat_tree(self):
        nodes = [
            self._node(1, None),
            self._node(2, None),
            self._node(3, None),
        ]
        counts = _compute_descendant_counts(nodes)
        assert counts[1] == 0
        assert counts[2] == 0
        assert counts[3] == 0

    def test_linear_chain(self):
        nodes = [
            self._node(1, None),
            self._node(2, 1),
            self._node(3, 2),
        ]
        counts = _compute_descendant_counts(nodes)
        assert counts[1] == 2  # 2 and 3
        assert counts[2] == 1  # 3
        assert counts[3] == 0

    def test_branching(self):
        nodes = [
            self._node(1, None),
            self._node(2, 1),
            self._node(3, 1),
            self._node(4, 2),
        ]
        counts = _compute_descendant_counts(nodes)
        assert counts[1] == 3  # 2, 3, 4
        assert counts[2] == 1  # 4
        assert counts[3] == 0
        assert counts[4] == 0

    def test_empty(self):
        counts = _compute_descendant_counts([])
        assert counts == {}
