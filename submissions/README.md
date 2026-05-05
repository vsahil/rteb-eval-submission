# RTEB Leaderboard Submissions

## What Submitters Provide

Each submission is a subdirectory containing:

1. **Model implementation** — a Python file that subclasses one of the base
   classes in `rteb/models/base.py`:
   - `SentenceTransformerModel` — for sentence-transformers compatible models
   - `OpenSourceModel` — for other open-source models
   - `APIModel` — for proprietary models behind an API

2. **Public-dataset results** — the JSON output from running on all public
   datasets (`python -m rteb.evaluate --output results/`).

3. **A brief model card** — model name, distance metric (default: cosine),
   and any special instructions for running the model.

## Expected Model Interface

All models must implement the following methods:

```python
class MyModel(OpenSourceModel):
    def __init__(self):
        super().__init__(model_name="my-model", distance_metric="cosine")

    def embed_query(self, query: str) -> list[float]:
        """Return the embedding vector for a single query."""
        ...

    def embed_document(self, document: str) -> list[float]:
        """Return the embedding vector for a single document."""
        ...
```

For batch-optimized models, override `embed_queries` and `embed_documents`:

```python
    def embed_queries(self, queries: list[str]) -> list[list[float]]:
        """Return embedding vectors for a batch of queries."""
        ...

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """Return embedding vectors for a batch of documents."""
        ...
```

## Private Evaluation

A large portion of RTEB datasets are kept private to prevent overfitting.
Private data is **not** included in this repository.

After submitting a pull request with the items above, the RTEB organizers will:

1. Run your model on all private datasets using the same evaluation pipeline.
2. Aggregate scores across public and private splits.
3. Update the leaderboard with the combined results.

Model providers without an explicit opt-out must sign a zero-day data retention
agreement to prevent data leakage.

## Rate Limits

Participants may submit at most **3 models per week** to prevent leaderboard
flooding.

## Directory Layout

```
submissions/
└── my-model/
    ├── model.py         # Model implementation
    ├── results.json     # Public-dataset results
    └── README.md        # Model card
```
