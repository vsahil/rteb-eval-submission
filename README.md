# RTEB: Retrieval Text Embedding Benchmark

RTEB is a retrieval-focused text embedding benchmark with private evaluation datasets to prevent overfitting. This repository contains the evaluation framework for benchmarking embedding models on RTEB.

## Key Features

- **Retrieval-focused**: Evaluates embedding models specifically on retrieval tasks
- **Private evaluation**: A large portion of datasets are kept private to prevent overfitting
- **Multi-domain**: Covers legal, finance, healthcare, code, and general domains
- **Multilingual**: English, German, Japanese, and French
- **Agentic code retrieval**: Dedicated evaluation for SWE-bench style issue-to-file retrieval

## Installation

```bash
# Clone the repository
git clone https://github.com/<anonymous>/rteb-eval.git
cd rteb-eval

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Quick Start

### 1. Evaluate a sentence-transformers model

```python
from rteb import RTEBEvaluator
from rteb.models import SentenceTransformerModel

# Load your model
model = SentenceTransformerModel("sentence-transformers/all-MiniLM-L6-v2")

# Run evaluation on all public datasets
evaluator = RTEBEvaluator()
results = evaluator.evaluate(model)

# Print results
evaluator.print_results(results)
```

### 2. Evaluate a custom open-source model

```python
from rteb.models import OpenSourceModel
import torch

class MyCustomModel(OpenSourceModel):
    def __init__(self):
        super().__init__(model_name="my-custom-model")
        # Load your model here
        self.tokenizer = ...
        self.model = ...

    def embed_query(self, query: str) -> list[float]:
        # Implement query embedding
        ...

    def embed_document(self, document: str) -> list[float]:
        # Implement document embedding
        ...

model = MyCustomModel()
evaluator = RTEBEvaluator()
results = evaluator.evaluate(model)
```

### 3. Evaluate a proprietary API model

```python
from rteb.models import APIModel

class MyAPIModel(APIModel):
    def __init__(self):
        super().__init__(model_name="my-api-model")
        self.api_key = "..."

    def embed_query(self, query: str) -> list[float]:
        # Call your API here
        ...

    def embed_document(self, document: str) -> list[float]:
        # Call your API here
        ...

model = MyAPIModel()
evaluator = RTEBEvaluator()
results = evaluator.evaluate(model)
```

### 4. Command-line evaluation

```bash
# Evaluate a sentence-transformers model
python -m rteb.evaluate \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --model-type sentence-transformer \
    --output results/

# Evaluate with BM25 baseline
python -m rteb.evaluate \
    --model bm25 \
    --model-type bm25 \
    --output results/

# Evaluate on specific datasets only
python -m rteb.evaluate \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --model-type sentence-transformer \
    --datasets AILA_casedocs LegalQuAD \
    --output results/

# Evaluate on specific domains only
python -m rteb.evaluate \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --model-type sentence-transformer \
    --domains legal healthcare \
    --output results/
```

### 5. Agentic-coding local evaluation

Agentic-coding tasks use the local format documented in
[`docs/data_format.md`](docs/data_format.md).

**Lexical baseline (smoke test, no model download required):**

The built-in `lexical` model type uses token-overlap cosine similarity. It
requires no GPU or external downloads and is intended only for verifying that the
evaluation pipeline runs end-to-end:

```bash
python -m rteb.evaluate_agentic_coding \
    --data-path tests/fixtures/agentic_coding_toy \
    --task-name toy_task \
    --model-type lexical \
    --output results/agentic_toy
```

**Sentence-transformer model (real evaluation):**

For meaningful retrieval scores, use `--model-type sentence-transformer` with
`--model-name` pointing to any sentence-transformers compatible model:

```bash
python -m rteb.evaluate_agentic_coding \
    --data-path /path/to/agentic_coding_data \
    --model-type sentence-transformer \
    --model-name sentence-transformers/all-MiniLM-L6-v2 \
    --output results/agentic_st
```

An environment-variable-based launch script is also provided:

```bash
DATA_PATH=/path/to/data OUTPUT_PATH=/tmp/out MODEL_TYPE=sentence-transformer \
    MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2 \
    bash scripts/run_agentic_coding_eval.sh
```

Both paths write `retrieve_pred.json`, `retrieve_eval.json`, and `summary.json`
per task.

> **Note:** The local evaluation path currently supports agentic-coding retrieval
> tasks only. Loaders for non-agentic text benchmarks will be added in a future
> release.

## Submitting to the RTEB Leaderboard

### Step 1: Implement your model

Inherit from one of the base classes in `rteb/models/`:
- `SentenceTransformerModel` — for sentence-transformers compatible models
- `OpenSourceModel` — for other open-source models
- `APIModel` — for proprietary models behind an API

### Step 2: Run evaluation on public datasets

```bash
python -m rteb.evaluate \
    --model <your-model> \
    --model-type <type> \
    --output results/
```

### Step 3: Submit a pull request

Your PR should include:
1. Your model implementation (in `submissions/`)
2. The distance metric to use (default: cosine similarity)
3. Results on all public datasets (in `results/`)

### Step 4: Private evaluation

The organizers will run your model on the private datasets and update the leaderboard.
To prevent data leakage, model providers without an explicit opt-out must sign a zero-day
data retention agreement.

### Rate Limits

Participants may submit at most 3 models per week to prevent leaderboard flooding.

## Verifying the Installation (Smoke Test)

After installing, you can verify that everything works with the included toy
fixture. No GPU, model downloads, or external data are required:

```bash
# Install in development mode
pip install -e ".[dev]"

# Run the test suite
pytest

# Run an end-to-end evaluation on the toy fixture
python -m rteb.evaluate_agentic_coding \
    --data-path tests/fixtures/agentic_coding_toy \
    --task-name toy_task \
    --model-type lexical \
    --output /tmp/rteb-agentic-toy
```

The last command should produce `retrieve_pred.json`, `retrieve_eval.json`, and
`summary.json` under `/tmp/rteb-agentic-toy/toy_task/`.

## Dataset Metadata

Croissant metadata files for each public dataset are provided in the
[`croissant/`](croissant/) directory. To validate a metadata file:

```bash
pip install mlcroissant
mlcroissant validate --jsonld croissant/<dataset>_croissant.json
```

## Repository Structure

```
rteb-eval/
├── LICENSE
├── README.md
├── requirements.txt
├── pyproject.toml
├── rteb/
│   ├── __init__.py
│   ├── evaluator.py                  # Main evaluation orchestrator
│   ├── evaluate.py                   # CLI: general evaluation
│   ├── evaluate_agentic_coding.py    # CLI: agentic-coding evaluation
│   ├── agentic_coding_dataset.py     # Local JSONL dataset loader
│   ├── metrics.py                    # NDCG, Recall implementations
│   ├── datasets.py                   # HuggingFace dataset loading
│   ├── retriever.py                  # Retrieval pipeline (embed + search)
│   └── models/
│       ├── __init__.py
│       ├── base.py                   # Abstract base classes
│       └── bm25.py                   # BM25 baseline
├── croissant/                        # Croissant metadata per dataset
├── docs/
│   └── data_format.md                # Task data format specification
├── examples/
│   ├── evaluate_sentence_transformer.py
│   ├── evaluate_custom_model.py
│   └── evaluate_bm25.py
├── scripts/
│   ├── download_datasets.py          # Download public datasets
│   └── run_agentic_coding_eval.sh    # Env-var launch script
├── submissions/                      # Leaderboard submission guide
│   └── README.md
└── tests/
    ├── test_agentic_coding_dataset.py
    ├── test_metrics.py
    ├── test_evaluate_agentic_coding.py
    └── fixtures/
        └── agentic_coding_toy/
```

## Datasets

The canonical local task format is documented in
[`docs/data_format.md`](docs/data_format.md). Public datasets can be evaluated
after conversion to this format; private datasets use the same abstraction but
are evaluated server-side by the RTEB organizers.

### Public Datasets

| Domain | Dataset | Language | Task |
|--------|---------|----------|------|
| General | FreshStack | English | Question → technical document |
| General | JaQuAD | Japanese | Question → passage |
| Legal | AILACasedocs | English | Scenario → case document |
| Legal | AILAStatutes | English | Scenario → statute |
| Legal | LegalSummarization | English | Excerpt → summary |
| Legal | LegalQuAD | German | Question → legal document |
| Finance | FinanceBench | English | Question → document excerpt |
| Finance | HC3Finance | English | Question → response |
| Finance | FinQA | English | Question → report evidence |
| Healthcare | ChatDoctor | English | Question → passage |
| Healthcare | HC3 Medicine | English | Question → response |
| Healthcare | HC3 French OOD | French | Question → response |
| Healthcare | Cure | English | Query → passage |
| Healthcare | TripClick | English | Search query → document |
| Code | HumanEval | English | Text → code |
| Code | MBPP | English | Text → code |
| Code | APPS | English | Text → code |
| Code | DS1000 | English | Text → code |
| Code | WikiSQL | English | Text → SQL |
| Agentic Code | Swe-bench++ | Multilingual | Issue → code |
| Agentic Code | Swe-bench-live | Multilingual | Issue → code |
| Agentic Code | Swe-Polybench | Multilingual | Issue → code |
| Agentic Code | Swing-bench | Multilingual | Issue → code |

### Private Datasets

28 private datasets across all domains. Names and details are withheld to preserve evaluation integrity. See the paper for domain-level statistics.

## Metrics

- **Primary**: NDCG@10 (displayed on leaderboard)
- **Additional**: {NDCG, Recall}@{1, 3, 5, 10, 20, 40, 100}


## License

This evaluation framework is released under the Apache 2.0 License. Individual datasets retain their original licenses.
