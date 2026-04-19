"""Tests for src.cluster: stance variance, cluster payloads."""

from __future__ import annotations

import json

import pytest

from src.cluster import _variance_of_stances, compute_cluster_payloads
from tests.conftest import seed_claim, seed_comment, seed_query, seed_thread



class TestVarianceOfStances:
    def test_uniform_pro(self):
        var = _variance_of_stances(["pro", "pro", "pro"])
        assert var == 0.0

    def test_mixed_pro_con(self):
        var = _variance_of_stances(["pro", "con"])
        assert var > 0

    def test_single_stance(self):
        var = _variance_of_stances(["pro"])
        assert var == 0.0

    def test_empty(self):
        var = _variance_of_stances([])
        assert var == 0.0

    def test_all_neutral(self):
        var = _variance_of_stances(["neutral", "neutral"])
        assert var == 0.0

    def test_diverse_stances(self):
        var = _variance_of_stances(["pro", "con", "alternative", "neutral"])
        assert var > 0



class TestComputeClusterPayloads:
    def test_basic_payload(self, tmp_db):
        import numpy as np

        qid = seed_query(tmp_db, "test topic")
        seed_thread(tmp_db, qid, 100, title="Test Thread", points=200)
        seed_comment(tmp_db, 1000, 100, depth=0, is_substantive=1)
        seed_comment(tmp_db, 1001, 100, depth=1, is_substantive=1)

        cid1 = seed_claim(tmp_db, 1000, claim_text="Claim 1", stance="pro", confidence=0.9)
        cid2 = seed_claim(tmp_db, 1001, claim_text="Claim 2", stance="pro", confidence=0.7)

        # Labels: both in cluster 0
        claim_ids = [cid1, cid2]
        labels = np.array([0, 0])

        payloads = compute_cluster_payloads(qid, claim_ids, labels)
        assert len(payloads) == 1
        assert payloads[0]["stance"] == "pro"
        assert len(payloads[0]["member_claim_ids"]) == 2
        assert payloads[0]["weight"] > 0

    def test_noise_cluster_excluded(self, tmp_db):
        import numpy as np

        qid = seed_query(tmp_db, "test")
        seed_thread(tmp_db, qid, 100, title="Thread")
        seed_comment(tmp_db, 1000, 100, depth=0, is_substantive=1)
        cid1 = seed_claim(tmp_db, 1000, claim_text="Noise claim", stance="pro")

        claim_ids = [cid1]
        labels = np.array([-1])  # noise

        payloads = compute_cluster_payloads(qid, claim_ids, labels)
        assert len(payloads) == 0

    def test_multiple_clusters(self, tmp_db):
        import numpy as np

        qid = seed_query(tmp_db, "test")
        seed_thread(tmp_db, qid, 100, title="Thread", points=300)
        seed_comment(tmp_db, 1000, 100, depth=0, is_substantive=1)
        seed_comment(tmp_db, 1001, 100, depth=0, is_substantive=1)

        cid1 = seed_claim(tmp_db, 1000, claim_text="Pro claim", stance="pro")
        cid2 = seed_claim(tmp_db, 1001, claim_text="Con claim", stance="con")

        labels = np.array([0, 1])
        payloads = compute_cluster_payloads(qid, [cid1, cid2], labels)
        assert len(payloads) == 2
        stances = {p["stance"] for p in payloads}
        assert "pro" in stances
        assert "con" in stances
