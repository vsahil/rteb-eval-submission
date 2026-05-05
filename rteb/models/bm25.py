"""
BM25 baseline for RTEB.

Unlike neural embedding models, BM25 does not produce embeddings.
It operates directly on the retrieval pipeline, so it has a custom
retrieve() method rather than embed_query/embed_document.
"""

from typing import Dict, List
from rank_bm25 import BM25Okapi
from tqdm import tqdm


class BM25Model:
    """
    BM25 lexical baseline.

    This model does not produce embeddings — it tokenizes and uses
    BM25Okapi for ranking. It is used directly with the evaluator
    via a custom retrieval path.

    Usage:
        model = BM25Model()
        evaluator = RTEBEvaluator()
        results = evaluator.evaluate(model)
    """

    def __init__(self, model_name: str = "BM25"):
        self.model_name = model_name
        self.distance_metric = "bm25"

    def retrieve(
        self,
        queries: Dict[str, str],
        corpus: Dict[str, str],
        top_k: int = 100,
        show_progress: bool = True,
    ) -> Dict[str, List[str]]:
        """
        Run BM25 retrieval.

        Args:
            queries: Dict mapping query_id -> query_text
            corpus: Dict mapping doc_id -> doc_text
            top_k: Number of top results per query

        Returns:
            Dict mapping query_id -> list of doc_ids (ranked by BM25 score)
        """
        # Build BM25 index
        doc_ids = list(corpus.keys())
        doc_texts = [corpus[did] for did in doc_ids]
        tokenized_corpus = [doc.lower().split() for doc in doc_texts]

        print("Building BM25 index...")
        bm25 = BM25Okapi(tokenized_corpus)

        # Retrieve for each query
        results = {}
        query_ids = list(queries.keys())
        iterator = tqdm(query_ids, desc="BM25 Retrieval", disable=not show_progress)

        for qid in iterator:
            query_tokens = queries[qid].lower().split()
            scores = bm25.get_scores(query_tokens)
            top_indices = scores.argsort()[::-1][:top_k]
            results[qid] = [doc_ids[idx] for idx in top_indices]

        return results

    def __repr__(self) -> str:
        return f"BM25Model(model_name='{self.model_name}')"
