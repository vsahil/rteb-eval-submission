import math

from rteb.metrics import K_VALUES, compute_metrics, ndcg_at_k, recall_at_k


def test_k_values_match_rteb_cutoffs():
    assert K_VALUES == [1, 3, 5, 10, 20, 50, 100]


def test_recall_at_k():
    retrieved = ["d2", "d1", "d3"]
    relevant = {"d1", "d3", "d4"}

    assert recall_at_k(retrieved, relevant, 1) == 0.0
    assert recall_at_k(retrieved, relevant, 2) == 1 / 3
    assert recall_at_k(retrieved, relevant, 3) == 2 / 3


def test_ndcg_uses_full_qrels_ideal_ranking():
    # Retrieved ranking only contains one relevant document, but the ideal
    # ranking has two relevant documents from qrels.
    actual_relevances = [1, 0]
    ideal_relevances = [1, 1]

    expected = 1.0 / (1.0 + 1.0 / math.log2(3))
    assert ndcg_at_k(actual_relevances, ideal_relevances, 2) == expected


def test_compute_metrics_names_and_values():
    query_results = {
        "q1": ["d2", "d1", "d3"],
        "q2": ["d4"],
    }
    qrels = {
        "q1": {"d1": 1, "d3": 1},
        "q2": {"d4": 2},
    }

    metrics = compute_metrics(query_results, qrels, k_values=[1, 3])

    assert set(metrics) == {
        "ndcg_at_1",
        "recall_at_1",
        "ndcg_at_3",
        "recall_at_3",
    }
    assert metrics["recall_at_1"] == 0.5
    assert metrics["recall_at_3"] == 1.0
    assert 0.0 < metrics["ndcg_at_1"] < 1.0
    expected_q1_ndcg_at_3 = (
        1.0 / math.log2(3) + 1.0 / math.log2(4)
    ) / (1.0 + 1.0 / math.log2(3))
    expected_q2_ndcg_at_3 = 1.0
    assert metrics["ndcg_at_3"] == (expected_q1_ndcg_at_3 + expected_q2_ndcg_at_3) / 2
