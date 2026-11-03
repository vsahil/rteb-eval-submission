"""
Command-line interface for RTEB evaluation.

Usage:
    python -m rteb.evaluate --model <model_name> --model-type <type> --output <dir>
"""

import argparse
import os

from rteb.evaluator import RTEBEvaluator
from rteb.datasets import list_datasets, list_domains


def main():
    parser = argparse.ArgumentParser(
        description="RTEB: Retrieval Text Embedding Benchmark Evaluation"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model name or path (e.g., 'sentence-transformers/all-MiniLM-L6-v2' or 'bm25')",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["sentence-transformer", "bm25"],
        help="Type of model to evaluate",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Specific dataset names to evaluate on (default: all public datasets)",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=None,
        help="Filter datasets by domain (e.g., 'legal' 'code')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/",
        help="Output directory for results (default: results/)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Number of top results to retrieve per query (default: 100)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for embedding (default: 64)",
    )
    parser.add_argument(
        "--distance-metric",
        type=str,
        default="cosine",
        choices=["cosine", "dot"],
        help="Distance metric for retrieval (default: cosine)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory for HuggingFace datasets",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List all available datasets and exit",
    )
    parser.add_argument(
        "--list-domains",
        action="store_true",
        help="List all available domains and exit",
    )

    args = parser.parse_args()

    # Handle list commands
    if args.list_datasets:
        print("Available RTEB datasets:")
        for name in list_datasets():
            print(f"  - {name}")
        return

    if args.list_domains:
        print("Available domains:")
        for domain in list_domains():
            datasets = list_datasets(domain)
            print(f"  {domain}: {len(datasets)} datasets")
            for name in datasets:
                print(f"    - {name}")
        return

    if not args.model or not args.model_type:
        parser.error("--model and --model-type are required for evaluation")

    # Load model
    if args.model_type == "sentence-transformer":
        from rteb.models import SentenceTransformerModel
        model = SentenceTransformerModel(
            args.model,
            distance_metric=args.distance_metric,
        )
    elif args.model_type == "bm25":
        from rteb.models import BM25Model
        model = BM25Model()
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    print(f"Model: {model}")
    print(f"Distance metric: {getattr(model, 'distance_metric', 'N/A')}")

    # Create evaluator
    evaluator = RTEBEvaluator(
        datasets=args.datasets,
        domains=args.domains,
        cache_dir=args.cache_dir,
        top_k=args.top_k,
        batch_size=args.batch_size,
    )

    print(f"Datasets to evaluate: {len(evaluator.dataset_names)}")

    # Run evaluation
    results = evaluator.evaluate(model, show_progress=True)

    # Print and save results
    evaluator.print_results(results)

    # Save results
    model_slug = args.model.replace("/", "_").replace(" ", "_")
    output_path = os.path.join(args.output, f"{model_slug}_results.json")
    evaluator.save_results(results, output_path, model_name=args.model)


if __name__ == "__main__":
    main()
