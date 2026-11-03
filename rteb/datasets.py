"""
Dataset loading and management for RTEB.

Handles downloading, caching, and formatting of RTEB public datasets.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from datasets import load_dataset as hf_load_dataset

# Registry of all public RTEB datasets
# Format: (dataset_name, hf_path, hf_subset, domain, language, query_field, doc_field, is_qa_repurposed)
DATASET_REGISTRY = {
    # General
    "FreshStack": {
        "hf_path": "freshstack/queries-oct-2024",
        "domain": "general",
        "language": "english",
        "qa_repurposed": False,
    },
    "JaQuAD": {
        "hf_path": "SkelterLabsInc/JaQuAD",
        "domain": "general",
        "language": "japanese",
        "qa_repurposed": True,
    },
    # Legal
    "AILA_casedocs": {
        "hf_path": "mteb/AILA_casedocs",
        "domain": "legal",
        "language": "english",
        "qa_repurposed": False,
    },
    "AILA_statutes": {
        "hf_path": "mteb/AILA_statutes",
        "domain": "legal",
        "language": "english",
        "qa_repurposed": False,
    },
    "legal_summarization": {
        "hf_path": "mteb/legal_summarization",
        "domain": "legal",
        "language": "english",
        "qa_repurposed": False,
    },
    "LegalQuAD": {
        "hf_path": "mteb/LegalQuAD",
        "domain": "legal",
        "language": "german",
        "qa_repurposed": False,
    },
    # Finance
    "financebench": {
        "hf_path": "virattt/financebench",
        "domain": "finance",
        "language": "english",
        "qa_repurposed": True,
    },
    "HC3Finance": {
        "hf_path": "Hello-SimpleAI/HC3",
        "hf_subset": "finance",
        "domain": "finance",
        "language": "english",
        "qa_repurposed": False,
    },
    "finqa": {
        "hf_path": "ibm/finqa",
        "domain": "finance",
        "language": "english",
        "qa_repurposed": True,
    },
    # Healthcare
    "ChatDoctor": {
        "hf_path": "lavita/ChatDoctor-HealthCareMagic-100k",
        "domain": "healthcare",
        "language": "english",
        "qa_repurposed": False,
    },
    "CUREv1": {
        "hf_path": "clinia/CUREv1",
        "domain": "healthcare",
        "language": "english",
        "qa_repurposed": False,
    },
    # Code
    "openai_humaneval": {
        "hf_path": "openai/openai_humaneval",
        "domain": "code",
        "language": "english",
        "qa_repurposed": True,
    },
    "mbpp": {
        "hf_path": "google-research-datasets/mbpp",
        "domain": "code",
        "language": "english",
        "qa_repurposed": True,
    },
    "apps": {
        "hf_path": "codeparrot/apps",
        "domain": "code",
        "language": "english",
        "qa_repurposed": True,
    },
    "DS-1000": {
        "hf_path": "xlangai/DS-1000",
        "domain": "code",
        "language": "english",
        "qa_repurposed": True,
    },
    "wikisql": {
        "hf_path": "Salesforce/wikisql",
        "domain": "code",
        "language": "english",
        "qa_repurposed": True,
    },
    # Agentic Code
    "swe-bench-plus-plus": {
        "hf_path": "TuringEnterprises/SWE-Bench-plus-plus",
        "domain": "agentic_code",
        "language": "multilingual",
        "qa_repurposed": False,
    },
    "swe-bench-live": {
        "hf_path": "SWE-bench-Live/SWE-bench-Live",
        "domain": "agentic_code",
        "language": "multilingual",
        "qa_repurposed": False,
    },
    "swe-polybench": {
        "hf_path": "AmazonScience/SWE-PolyBench",
        "domain": "agentic_code",
        "language": "multilingual",
        "qa_repurposed": False,
    },
    "swing-bench": {
        "hf_path": "SwingBench/SwingBench",
        "domain": "agentic_code",
        "language": "multilingual",
        "qa_repurposed": False,
    },
}

DOMAINS = ["general", "legal", "finance", "healthcare", "code", "agentic_code"]


@dataclass
class RTEBDataset:
    """Represents a single RTEB dataset ready for evaluation."""

    name: str
    domain: str
    language: str
    queries: Dict[str, str] = field(default_factory=dict)  # query_id -> query_text
    corpus: Dict[str, str] = field(default_factory=dict)  # doc_id -> doc_text
    qrels: Dict[str, Dict[str, int]] = field(
        default_factory=dict
    )  # query_id -> {doc_id: relevance}

    @property
    def num_queries(self) -> int:
        return len(self.queries)

    @property
    def num_documents(self) -> int:
        return len(self.corpus)

    def __repr__(self) -> str:
        return (
            f"RTEBDataset(name='{self.name}', domain='{self.domain}', "
            f"language='{self.language}', queries={self.num_queries}, "
            f"documents={self.num_documents})"
        )


def load_dataset(
    dataset_name: str,
    cache_dir: Optional[str] = None,
) -> RTEBDataset:
    """
    Load a single RTEB dataset by name.

    Args:
        dataset_name: Name of the dataset (must be in DATASET_REGISTRY)
        cache_dir: Optional cache directory for HuggingFace datasets

    Returns:
        RTEBDataset with queries, corpus, and qrels populated

    Raises:
        ValueError: If dataset_name is not in the registry
    """
    if dataset_name not in DATASET_REGISTRY:
        available = ", ".join(sorted(DATASET_REGISTRY.keys()))
        raise ValueError(
            f"Unknown dataset: '{dataset_name}'. Available datasets: {available}"
        )

    config = DATASET_REGISTRY[dataset_name]
    hf_path = config["hf_path"]
    hf_subset = config.get("hf_subset", None)

    print(f"Loading dataset: {dataset_name} from {hf_path}")

    # Load from HuggingFace
    kwargs = {"path": hf_path}
    if hf_subset:
        kwargs["name"] = hf_subset
    if cache_dir:
        kwargs["cache_dir"] = cache_dir

    raw_data = hf_load_dataset(**kwargs)

    # Parse into RTEB format
    # NOTE: The exact parsing logic depends on each dataset's schema.
    # This is a general framework — dataset-specific adapters may be needed.
    # The adapters are registered below and handle the conversion from
    # each dataset's raw format to the standardized (queries, corpus, qrels) format.

    dataset = RTEBDataset(
        name=dataset_name,
        domain=config["domain"],
        language=config["language"],
    )

    adapter = get_adapter(dataset_name)
    dataset.queries, dataset.corpus, dataset.qrels = adapter(raw_data)

    print(
        f"  Loaded: {dataset.num_queries} queries, "
        f"{dataset.num_documents} documents"
    )

    return dataset


def get_adapter(dataset_name: str):
    """
    Get the dataset-specific adapter function.

    Each adapter converts a HuggingFace dataset into the standardized
    (queries, corpus, qrels) format used by RTEB.

    Returns a callable: raw_data -> (queries, corpus, qrels)
    """
    # Dataset-specific adapters
    adapters = {
        "AILA_casedocs": _adapter_mteb_retrieval,
        "AILA_statutes": _adapter_mteb_retrieval,
        "legal_summarization": _adapter_mteb_retrieval,
        "LegalQuAD": _adapter_mteb_retrieval,
        # Add more dataset-specific adapters here as needed.
        # Datasets without a specific adapter use the default adapter.
    }

    return adapters.get(dataset_name, _adapter_default)


def _adapter_mteb_retrieval(raw_data) -> Tuple[Dict, Dict, Dict]:
    """
    Adapter for datasets in MTEB/BEIR retrieval format.
    Expected splits: 'queries', 'corpus', and qrels as a mapping.
    """
    queries = {}
    corpus = {}
    qrels = {}

    # Load queries
    if "queries" in raw_data:
        for row in raw_data["queries"]:
            queries[str(row["_id"])] = row["text"]

    # Load corpus
    if "corpus" in raw_data:
        for row in raw_data["corpus"]:
            doc_text = row.get("text", "")
            doc_title = row.get("title", "")
            if doc_title:
                doc_text = f"{doc_title} {doc_text}"
            corpus[str(row["_id"])] = doc_text

    # Load qrels (relevance judgments)
    if "default" in raw_data:
        for row in raw_data["default"]:
            qid = str(row["query-id"])
            did = str(row["corpus-id"])
            rel = int(row["score"])
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][did] = rel

    return queries, corpus, qrels


def _adapter_default(raw_data) -> Tuple[Dict, Dict, Dict]:
    """
    Default adapter that attempts to parse common dataset formats.
    This is a fallback — dataset-specific adapters should be preferred.
    """
    queries = {}
    corpus = {}
    qrels = {}

    # Try to find the main split
    split_name = None
    for candidate in ["test", "validation", "train"]:
        if candidate in raw_data:
            split_name = candidate
            break

    if split_name is None:
        split_name = list(raw_data.keys())[0]

    data = raw_data[split_name]

    # Attempt to identify query and document fields
    columns = data.column_names
    query_field = None
    doc_field = None

    # Common query field names
    for candidate in ["question", "query", "input", "prompt"]:
        if candidate in columns:
            query_field = candidate
            break

    # Common document field names
    for candidate in ["context", "answer", "passage", "document", "text", "output"]:
        if candidate in columns:
            doc_field = candidate
            break

    if query_field is None or doc_field is None:
        raise ValueError(
            f"Could not auto-detect query/document fields from columns: {columns}. "
            f"Please implement a dataset-specific adapter."
        )

    # Build queries, corpus, and qrels
    for idx, row in enumerate(data):
        qid = f"q_{idx}"
        did = f"d_{idx}"
        queries[qid] = str(row[query_field])
        corpus[did] = str(row[doc_field])
        qrels[qid] = {did: 1}

    return queries, corpus, qrels


def list_datasets(domain: Optional[str] = None) -> List[str]:
    """
    List available datasets, optionally filtered by domain.

    Args:
        domain: Optional domain filter (e.g., 'legal', 'code')

    Returns:
        List of dataset names
    """
    if domain is None:
        return sorted(DATASET_REGISTRY.keys())
    return sorted(
        name
        for name, config in DATASET_REGISTRY.items()
        if config["domain"] == domain
    )


def list_domains() -> List[str]:
    """Return list of available domains."""
    return DOMAINS.copy()
