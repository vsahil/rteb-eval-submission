"""
Example: Evaluate a custom open-source model on RTEB.

This demonstrates how to wrap any model using the OpenSourceModel base class.

Usage:
    python examples/evaluate_custom_model.py
"""

from typing import List

import torch
from transformers import AutoTokenizer, AutoModel

from rteb import RTEBEvaluator
from rteb.models import OpenSourceModel


class CustomBERTModel(OpenSourceModel):
    """Example custom model wrapping a BERT-based encoder."""

    def __init__(self, model_name: str = "bert-base-uncased"):
        super().__init__(model_name=model_name, distance_metric="cosine")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def _encode(self, text: str) -> List[float]:
        """Encode a single text using mean pooling."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Mean pooling over token embeddings
        attention_mask = inputs["attention_mask"]
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        embedding = torch.sum(
            token_embeddings * input_mask_expanded, 1
        ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        # Normalize
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        return embedding.squeeze().tolist()

    def embed_query(self, query: str) -> List[float]:
        return self._encode(query)

    def embed_document(self, document: str) -> List[float]:
        return self._encode(document)


def main():
    # Load custom model
    model = CustomBERTModel("bert-base-uncased")

    # Evaluate on a subset of datasets
    evaluator = RTEBEvaluator(
        datasets=["AILA_casedocs"],
    )

    # Run evaluation
    results = evaluator.evaluate(model)

    # Print results
    evaluator.print_results(results)

    # Save results
    evaluator.save_results(
        results,
        output_path="results/bert-base_results.json",
        model_name="bert-base-uncased",
    )


if __name__ == "__main__":
    main()
