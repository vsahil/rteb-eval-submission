"""
RTEB Evaluator: orchestrates dataset loading, retrieval, and metric computation.
"""

import json
import os
from typing import Dict, List, Optional

import pandas as pd

from rteb.datasets import RTEBDataset, load_dataset, list_datasets, list_domains
from rteb.metrics import compute_metrics, K_VALUES
from rteb.retriever import retrieve
from rteb.models.bm25 import BM25Model


class RTEBEvaluator:
    """
    Main evaluation orchestrator for RTEB.

    Usage:
        evaluator = RTEBEvaluator()
        model = SentenceTransformerModel("all-MiniLM-L6-v2")
        results = evaluator.evaluate(model)
        evaluator.print_results(results)
        evaluator.save_results(results, "results/my_model.json")
    """

    def __init__(
        self,
        datasets: Optional[List[str]] = None,
        domains: Optional[List[str]] = None,
        cache_dir: Optional[str] = None,
        top_k: int = 100,
        batch_size: int = 64,
    ):
        """
        Args:
            datasets: List of dataset names to evaluate on. If None, uses all public datasets.
            domains: List of domains to filter by. Ignored if datasets is specified.
            cache_dir: Cache directory for HuggingFace datasets.
            top_k: Number of top results to retrieve per query.
            batch_size: Batch size for embedding.
        """
        self.cache_dir = cache_dir
        self.top_k = top_k
        self.batch_size = batch_size

        # Determine which datasets to evaluate
        if datasets is not None:
            self.dataset_names = datasets
        elif domains is not None:
            self.dataset_names = []
            for domain in domains:
                self.dataset_names.extend(list_datasets(domain))
        else:
            self.dataset_names = list_datasets()

    def evaluate(
        self,
        model,
        show_progress: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate a model on all configured datasets.

        Args:
            model: An embedding model (BaseModel subclass) or BM25Model
            show_progress: Whether to show progress bars

        Returns:
            Dict mapping dataset_name -> {metric_name: score}
        """
        all_results = {}

        for dataset_name in self.dataset_names:
            print(f"\n{'='*60}")
            print(f"Evaluating on: {dataset_name}")
            print(f"{'='*60}")

            # Load dataset
            try:
                dataset = load_dataset(dataset_name, cache_dir=self.cache_dir)
            except Exception as e:
                print(f"  ERROR loading {dataset_name}: {e}")
                continue

            # Run retrieval
            if isinstance(model, BM25Model):
                query_results = model.retrieve(
                    queries=dataset.queries,
                    corpus=dataset.corpus,
                    top_k=self.top_k,
                    show_progress=show_progress,
                )
            else:
                query_results = retrieve(
                    model=model,
                    queries=dataset.queries,
                    corpus=dataset.corpus,
                    top_k=self.top_k,
                    distance_metric=model.distance_metric,
                    batch_size=self.batch_size,
                    show_progress=show_progress,
                )

            # Compute metrics
            metrics = compute_metrics(
                query_results=query_results,
                qrels=dataset.qrels,
                k_values=K_VALUES,
            )

            all_results[dataset_name] = metrics
            print(f"  NDCG@10: {metrics.get('ndcg_at_10', 0.0):.4f}")

        return all_results

    def print_results(self, results: Dict[str, Dict[str, float]]) -> None:
        """Print results as a formatted table."""
        if not results:
            print("No results to display.")
            return

        print(f"\n{'='*80}")
        print("RTEB Evaluation Results")
        print(f"{'='*80}")

        # Build table data
        rows = []
        for dataset_name, metrics in sorted(results.items()):
            row = {"Dataset": dataset_name}
            for k in K_VALUES:
                row[f"NDCG@{k}"] = metrics.get(f"ndcg_at_{k}", 0.0)
                row[f"Recall@{k}"] = metrics.get(f"recall_at_{k}", 0.0)
            rows.append(row)

        df = pd.DataFrame(rows)

        # Print NDCG table
        ndcg_cols = ["Dataset"] + [f"NDCG@{k}" for k in K_VALUES]
        print("\nNDCG Scores:")
        print(df[ndcg_cols].to_string(index=False, float_format="%.4f"))

        # Print Recall table
        recall_cols = ["Dataset"] + [f"Recall@{k}" for k in K_VALUES]
        print("\nRecall Scores:")
        print(df[recall_cols].to_string(index=False, float_format="%.4f"))

        # Print aggregate
        print(f"\n{'='*80}")
        print("Aggregate (mean across datasets):")
        for k in K_VALUES:
            ndcg_mean = df[f"NDCG@{k}"].mean()
            recall_mean = df[f"Recall@{k}"].mean()
            print(f"  NDCG@{k}: {ndcg_mean:.4f}  |  Recall@{k}: {recall_mean:.4f}")

    def save_results(
        self,
        results: Dict[str, Dict[str, float]],
        output_path: str,
        model_name: Optional[str] = None,
    ) -> None:
        """
        Save results to a JSON file.

        Args:
            results: Evaluation results from evaluate()
            output_path: Path to save the JSON file
            model_name: Optional model name to include in the output
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        output = {
            "model_name": model_name,
            "num_datasets": len(results),
            "results": results,
        }

        # Add aggregate metrics
        if results:
            aggregate = {}
            for k in K_VALUES:
                ndcg_scores = [
                    m.get(f"ndcg_at_{k}", 0.0) for m in results.values()
                ]
                recall_scores = [
                    m.get(f"recall_at_{k}", 0.0) for m in results.values()
                ]
                aggregate[f"ndcg_at_{k}"] = (
                    sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0
                )
                aggregate[f"recall_at_{k}"] = (
                    sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
                )
            output["aggregate"] = aggregate

        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"\nResults saved to: {output_path}")
