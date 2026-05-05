"""
Retrieval pipeline: embed queries and documents, then rank by similarity.
"""

import numpy as np
from typing import Dict, List, Optional
from tqdm import tqdm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between two matrices."""
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return a_norm @ b_norm.T


def dot_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute dot product similarity between two matrices."""
    return a @ b.T


DISTANCE_METRICS = {
    "cosine": cosine_similarity,
    "dot": dot_product,
}


def retrieve(
    model,
    queries: Dict[str, str],
    corpus: Dict[str, str],
    top_k: int = 100,
    distance_metric: str = "cosine",
    batch_size: int = 64,
    show_progress: bool = True,
) -> Dict[str, List[str]]:
    """
    Run retrieval: embed queries and corpus, compute similarities, return ranked results.

    Args:
        model: An embedding model with embed_query() and embed_document() methods,
               or embed_queries() and embed_documents() for batch processing.
        queries: Dict mapping query_id -> query_text
        corpus: Dict mapping doc_id -> doc_text
        top_k: Number of top results to return per query
        distance_metric: 'cosine' or 'dot'
        batch_size: Batch size for embedding
        show_progress: Whether to show progress bars

    Returns:
        Dict mapping query_id -> list of doc_ids (ranked by similarity)
    """
    if distance_metric not in DISTANCE_METRICS:
        raise ValueError(
            f"Unknown distance metric: '{distance_metric}'. "
            f"Available: {list(DISTANCE_METRICS.keys())}"
        )

    similarity_fn = DISTANCE_METRICS[distance_metric]

    # Embed corpus
    doc_ids = list(corpus.keys())
    doc_texts = [corpus[did] for did in doc_ids]
    doc_embeddings = _embed_batch(
        model, doc_texts, mode="document", batch_size=batch_size,
        desc="Embedding corpus" if show_progress else None,
    )

    # Embed queries
    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]
    query_embeddings = _embed_batch(
        model, query_texts, mode="query", batch_size=batch_size,
        desc="Embedding queries" if show_progress else None,
    )

    # Compute similarities and rank
    results = {}
    for i, qid in enumerate(
        tqdm(query_ids, desc="Ranking", disable=not show_progress)
    ):
        q_emb = query_embeddings[i : i + 1]
        scores = similarity_fn(q_emb, doc_embeddings)[0]
        top_indices = np.argsort(scores)[::-1][:top_k]
        results[qid] = [doc_ids[idx] for idx in top_indices]

    return results


def _embed_batch(
    model,
    texts: List[str],
    mode: str,
    batch_size: int,
    desc: Optional[str] = None,
) -> np.ndarray:
    """
    Embed a list of texts in batches.

    Supports both single-text methods (embed_query/embed_document)
    and batch methods (embed_queries/embed_documents).
    """
    # Check for batch methods first
    if mode == "query" and hasattr(model, "embed_queries"):
        return np.array(model.embed_queries(texts))
    if mode == "document" and hasattr(model, "embed_documents"):
        return np.array(model.embed_documents(texts))

    # Fall back to single-text methods
    embed_fn = model.embed_query if mode == "query" else model.embed_document

    embeddings = []
    iterator = range(0, len(texts), batch_size)
    if desc:
        iterator = tqdm(iterator, desc=desc)

    for start in iterator:
        batch = texts[start : start + batch_size]
        batch_embeddings = [embed_fn(text) for text in batch]
        embeddings.extend(batch_embeddings)

    return np.array(embeddings)
