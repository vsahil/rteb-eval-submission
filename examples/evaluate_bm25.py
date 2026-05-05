"""
Example: Evaluate BM25 baseline on RTEB.

Usage:
    python examples/evaluate_bm25.py
"""

from rteb import RTEBEvaluator
from rteb.models import BM25Model


def main():
    # BM25 baseline
    model = BM25Model()

    # Evaluate on a subset of datasets (for quick testing)
    evaluator = RTEBEvaluator(
        datasets=["AILA_casedocs", "LegalQuAD"],
    )

    # Run evaluation
    results = evaluator.evaluate(model)

    # Print results
    evaluator.print_results(results)

    # Save results
    evaluator.save_results(
        results,
        output_path="results/bm25_results.json",
        model_name="BM25",
    )


if __name__ == "__main__":
    main()
