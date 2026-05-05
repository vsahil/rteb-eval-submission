"""
CLI for evaluating local RTEB agentic-coding retrieval tasks.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List

from rteb.agentic_coding_dataset import (
    list_agentic_coding_tasks,
    load_agentic_coding_task,
)
from rteb.metrics import K_VALUES, compute_metrics
from rteb.retriever import retrieve


def evaluate_agentic_coding_tasks(
    *,
    data_path: str | Path,
    task_names: List[str],
    model,
    output_path: str | Path,
    top_k: int = 100,
    batch_size: int = 64,
    add_instructions: bool = False,
    show_progress: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Evaluate a model on local agentic-coding tasks and save per-task outputs."""
    output_root = Path(output_path)
    output_root.mkdir(parents=True, exist_ok=True)

    all_results: Dict[str, Dict[str, float]] = {}
    for task_name in task_names:
        dataset = load_agentic_coding_task(
            data_path,
            task_name,
            add_instructions=add_instructions,
        )

        if isinstance(model, LexicalOverlapModel):
            ranked = model.retrieve(
                dataset.queries,
                dataset.corpus,
                top_k=top_k,
                show_progress=show_progress,
            )
        else:
            ranked = retrieve(
                model=model,
                queries=dataset.queries,
                corpus=dataset.corpus,
                top_k=top_k,
                distance_metric=model.distance_metric,
                batch_size=batch_size,
                show_progress=show_progress,
            )

        prediction = _ranked_ids_to_scored_prediction(ranked, top_k=top_k)
        metrics = compute_metrics(ranked, dataset.relevance, k_values=K_VALUES)
        all_results[task_name] = metrics

        task_output = output_root / task_name
        task_output.mkdir(parents=True, exist_ok=True)
        _write_json(task_output / "retrieve_pred.json", prediction)
        _write_json(task_output / "retrieve_eval.json", metrics)

        print(f"{task_name}: ndcg_at_10={metrics.get('ndcg_at_10', 0.0):.4f}")

    _write_json(output_root / "summary.json", all_results)
    return all_results


class LexicalOverlapModel:
    """Small dependency-free lexical baseline for smoke tests and examples."""

    model_name = "lexical-overlap"
    distance_metric = "lexical-overlap"

    def retrieve(
        self,
        queries: Dict[str, str],
        corpus: Dict[str, str],
        top_k: int = 100,
        show_progress: bool = True,
    ) -> Dict[str, List[str]]:
        del show_progress
        doc_tokens = {
            doc_id: Counter(_tokenize(text))
            for doc_id, text in corpus.items()
        }
        doc_norms = {
            doc_id: math.sqrt(sum(count * count for count in counts.values()))
            for doc_id, counts in doc_tokens.items()
        }

        rankings: Dict[str, List[str]] = {}
        for query_id, query in queries.items():
            query_counts = Counter(_tokenize(query))
            query_norm = math.sqrt(sum(count * count for count in query_counts.values()))
            scored = []
            for doc_id, counts in doc_tokens.items():
                score = 0.0
                if query_norm and doc_norms[doc_id]:
                    overlap = set(query_counts) & set(counts)
                    dot = sum(query_counts[token] * counts[token] for token in overlap)
                    score = dot / (query_norm * doc_norms[doc_id])
                scored.append((score, doc_id))
            scored.sort(key=lambda item: (-item[0], item[1]))
            rankings[query_id] = [doc_id for _, doc_id in scored[:top_k]]
        return rankings


def _tokenize(text: str) -> List[str]:
    return text.lower().replace("/", " ").replace("_", " ").replace("-", " ").split()


def _ranked_ids_to_scored_prediction(
    ranked: Dict[str, List[str]],
    top_k: int,
) -> Dict[str, Dict[str, float]]:
    prediction: Dict[str, Dict[str, float]] = {}
    for query_id, doc_ids in ranked.items():
        prediction[query_id] = {
            doc_id: float(top_k - rank)
            for rank, doc_id in enumerate(doc_ids[:top_k])
        }
    return prediction


def _write_json(path: Path, data) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def _load_model(args):
    if args.model_type == "lexical":
        return LexicalOverlapModel()
    if args.model_type == "sentence-transformer":
        from rteb.models.base import SentenceTransformerModel

        return SentenceTransformerModel(
            args.model_name,
            distance_metric=args.distance_metric,
        )
    raise ValueError(f"Unsupported model type: {args.model_type}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate local RTEB agentic-coding retrieval tasks."
    )
    parser.add_argument("--data-path", required=True, help="Root directory of local tasks.")
    parser.add_argument(
        "--task-name",
        nargs="+",
        default=None,
        help="Task name(s). If omitted, all local agentic-coding tasks are used.",
    )
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument(
        "--model-type",
        default="lexical",
        choices=["lexical", "sentence-transformer"],
    )
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--distance-metric", default="cosine", choices=["cosine", "dot"])
    parser.add_argument("--add-instructions", action="store_true")
    args = parser.parse_args()

    if args.task_name:
        task_names = args.task_name
    else:
        task_names = list_agentic_coding_tasks(args.data_path)
    if not task_names:
        raise ValueError(f"No agentic-coding tasks found under {args.data_path}")

    if args.model_type == "sentence-transformer" and not args.model_name:
        raise ValueError("--model-name is required for sentence-transformer models")

    model = _load_model(args)
    os.makedirs(args.output, exist_ok=True)
    evaluate_agentic_coding_tasks(
        data_path=args.data_path,
        task_names=task_names,
        model=model,
        output_path=args.output,
        top_k=args.top_k,
        batch_size=args.batch_size,
        add_instructions=args.add_instructions,
    )


if __name__ == "__main__":
    main()
