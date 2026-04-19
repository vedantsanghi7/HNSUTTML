"""Tests for src.extract: signal score, validation, message building."""

from __future__ import annotations

import pytest

from src.extract import (
    ClaimExtraction,
    _build_user_message,
    _signal_score,
    _validate,
)



class TestSignalScore:
    def _row(self, depth=0, text_length=200, descendant_count=5, has_code=0):
        # Simulate a sqlite3.Row with dict-like access.
        return {
            "depth": depth,
            "text_length": text_length,
            "descendant_count": descendant_count,
            "has_code": has_code,
        }

    def test_basic_positive(self):
        score = _signal_score(self._row())
        assert score > 0

    def test_code_bonus(self):
        without_code = _signal_score(self._row(has_code=0))
        with_code = _signal_score(self._row(has_code=1))
        assert with_code > without_code

    def test_longer_text_better(self):
        short = _signal_score(self._row(text_length=50))
        long = _signal_score(self._row(text_length=1000))
        assert long > short

    def test_deep_penalty(self):
        # Comments deeper than 5 should get a penalty.
        shallow = _signal_score(self._row(depth=3))
        deep = _signal_score(self._row(depth=8))
        assert shallow > deep

    def test_more_descendants_better(self):
        few = _signal_score(self._row(descendant_count=1))
        many = _signal_score(self._row(descendant_count=50))
        assert many > few



class TestValidate:
    def test_valid_extraction(self):
        args = {
            "is_substantive": True,
            "claims": [
                {
                    "text": "SQLite handles 1M reads per second",
                    "stance": "benchmark",
                    "category": "performance",
                    "evidence_type": "benchmark",
                    "tools_mentioned": ["SQLite"],
                    "confidence": 0.9,
                    "is_firsthand": False,
                }
            ],
        }
        result = _validate(args)
        assert result is not None
        assert result.is_substantive is True
        assert len(result.claims) == 1

    def test_empty_claims(self):
        args = {"is_substantive": False, "claims": []}
        result = _validate(args)
        assert result is not None
        assert result.claims == []

    def test_none_input(self):
        assert _validate(None) is None

    def test_invalid_stance(self):
        args = {
            "is_substantive": True,
            "claims": [
                {
                    "text": "test",
                    "stance": "invalid_stance",
                    "category": "test",
                    "evidence_type": "opinion",
                    "tools_mentioned": [],
                    "confidence": 0.5,
                    "is_firsthand": False,
                }
            ],
        }
        assert _validate(args) is None

    def test_confidence_out_of_range(self):
        args = {
            "is_substantive": True,
            "claims": [
                {
                    "text": "test",
                    "stance": "pro",
                    "category": "test",
                    "evidence_type": "opinion",
                    "tools_mentioned": [],
                    "confidence": 1.5,  # out of [0, 1]
                    "is_firsthand": False,
                }
            ],
        }
        assert _validate(args) is None

    def test_missing_required_field(self):
        args = {
            "is_substantive": True,
            "claims": [
                {
                    "text": "test",
                    # missing stance, category, etc.
                }
            ],
        }
        assert _validate(args) is None



class TestBuildUserMessage:
    def test_basic_format(self):
        msg = _build_user_message(42, "Responding to parent", "This is my comment")
        assert "Context: Responding to parent" in msg
        assert '<hn_comment id="42">' in msg
        assert "This is my comment" in msg
        assert "</hn_comment>" in msg

    def test_empty_prefix(self):
        msg = _build_user_message(1, "", "Comment text")
        assert "Context:" in msg
        assert '<hn_comment id="1">' in msg

    def test_newlines_stripped_from_prefix(self):
        msg = _build_user_message(1, "line1\nline2\nline3", "text")
        # The prefix should have newlines replaced with spaces
        assert "line1 line2 line3" in msg
