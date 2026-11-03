"""
Evaluation metrics for RTEB: NDCG and Recall at various cutoffs.
"""

import numpy as np
from typing import Dict, List, Optional

K_VALUES = [1, 3, 5, 10, 20, 50, 100]


def dcg_at_k(relevances: List[int], k: int) -> float:
    """Compute Discounted Cumulative Gain at rank k."""
    relevances = np.array(relevances[:k])
    if relevances.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, relevances.size + 2))
    return float(np.sum(relevances / discounts))


def ndcg_at_k(relevances: List[int], ideal_relevances: List[int], k: int) -> float:
    """Compute Normalized Discounted Cumulative Gain at rank k."""
    dcg = dcg_at_k(relevances, k)
    idcg = dcg_at_k(ideal_relevances, k)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def recall_at_k(retrieved_ids: List[str], relevant_ids: set, k: int) -> float:
    """Compute Recall at rank k."""
    if len(relevant_ids) == 0:
        return 0.0
    retrieved_at_k = set(retrieved_ids[:k])
    return len(retrieved_at_k & relevant_ids) / len(relevant_ids)


def compute_metrics(
    query_results: Dict[str, List[str]],
    qrels: Dict[str, Dict[str, int]],
    k_values: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    Compute NDCG and Recall at various k values.

    Args:
        query_results: Dict mapping query_id -> list of retrieved document_ids (ranked)
        qrels: Dict mapping query_id -> {doc_id: relevance_score}
        k_values: List of k values to compute metrics at. Defaults to
            [1, 3, 5, 10, 20, 50, 100].

    Returns:
        Dict mapping metric_name -> score (averaged across queries)
    """
    if k_values is None:
        k_values = K_VALUES

    metrics = {}
    for k in k_values:
        metrics[f"ndcg_at_{k}"] = []
        metrics[f"recall_at_{k}"] = []

    # Every judged query contributes, including queries with no predictions.
    for query_id, query_rels in qrels.items():
        ranked_doc_ids = query_results.get(query_id, [])
        relevant_ids = {doc_id for doc_id, rel in query_rels.items() if rel > 0}
        ideal_relevances = sorted(
            (rel for rel in query_rels.values() if rel > 0),
            reverse=True,
        )

        # Build relevance list in retrieval order
        relevances = [query_rels.get(doc_id, 0) for doc_id in ranked_doc_ids]

        for k in k_values:
            metrics[f"ndcg_at_{k}"].append(ndcg_at_k(relevances, ideal_relevances, k))
            metrics[f"recall_at_{k}"].append(recall_at_k(ranked_doc_ids, relevant_ids, k))

    # Average across queries
    averaged = {}
    for metric_name, scores in metrics.items():
        if len(scores) > 0:
            averaged[metric_name] = float(np.mean(scores))
        else:
            averaged[metric_name] = 0.0

    return averaged
