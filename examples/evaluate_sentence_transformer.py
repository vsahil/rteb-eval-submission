"""
Example: Evaluate a sentence-transformers model on RTEB.

Usage:
    python examples/evaluate_sentence_transformer.py
"""

from rteb import RTEBEvaluator
from rteb.models import SentenceTransformerModel


def main():
    # Load model
    model = SentenceTransformerModel(
        model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
        distance_metric="cosine",
    )

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
        output_path="results/all-MiniLM-L6-v2_results.json",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )


if __name__ == "__main__":
    main()
