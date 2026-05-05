"""
Local dataset loader for RTEB agentic coding retrieval tasks.

This module intentionally handles only the local agentic-coding task format used
for issue-to-code retrieval. Normal text benchmark adapters are separate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


DEFAULT_QUERY_INSTRUCTION = "Represent the query for retrieving supporting documents: "
DEFAULT_CORPUS_INSTRUCTION = "Represent the document for retrieval: "


@dataclass
class AgenticCodingDataset:
    """Agentic-code retrieval task loaded from local files."""

    name: str
    task_path: Path
    queries: Dict[str, str] = field(default_factory=dict)
    corpus: Dict[str, str] = field(default_factory=dict)
    relevance: Dict[str, Dict[str, int]] = field(default_factory=dict)
    query_instruction: str = DEFAULT_QUERY_INSTRUCTION
    corpus_instruction: str = DEFAULT_CORPUS_INSTRUCTION

    @property
    def qrels(self) -> Dict[str, Dict[str, int]]:
        """Alias used by the generic evaluator."""
        return self.relevance

    @property
    def num_queries(self) -> int:
        return len(self.queries)

    @property
    def num_documents(self) -> int:
        return len(self.corpus)


def load_agentic_coding_task(
    data_path: str | Path,
    task_name: str,
    *,
    add_instructions: bool = False,
) -> AgenticCodingDataset:
    """
    Load one local agentic-coding retrieval task.

    Expected files:
      <data_path>/<task_name>/queries.jsonl
      <data_path>/<task_name>/corpus.jsonl
      <data_path>/<task_name>/relevance.jsonl or relevance.json
      <data_path>/<task_name>/custom_instruction.json (optional)
    """
    root = Path(data_path)
    task_path = root / task_name
    if not task_path.is_dir():
        raise FileNotFoundError(f"Agentic coding task directory not found: {task_path}")

    instructions = _load_custom_instructions(task_path)
    dataset = AgenticCodingDataset(
        name=task_name,
        task_path=task_path,
        query_instruction=instructions["query"],
        corpus_instruction=instructions["corpus"],
    )

    dataset.queries = _load_text_records(
        task_path / "queries.jsonl",
        instruction=dataset.query_instruction if add_instructions else "",
    )
    dataset.corpus = _load_text_records(
        task_path / "corpus.jsonl",
        instruction=dataset.corpus_instruction if add_instructions else "",
    )
    dataset.relevance = _load_relevance(task_path)

    _validate_agentic_coding_dataset(dataset)
    return dataset


def list_agentic_coding_tasks(data_path: str | Path) -> list[str]:
    """List local task directories that look like agentic-coding retrieval tasks."""
    root = Path(data_path)
    if not root.is_dir():
        raise FileNotFoundError(f"Data path not found: {root}")

    tasks = []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        if (path / "queries.jsonl").exists() and (path / "corpus.jsonl").exists():
            tasks.append(path.name)
    return tasks


def _load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object in {path}:{line_number}")
            yield value


def _load_text_records(path: Path, instruction: str = "") -> Dict[str, str]:
    records: Dict[str, str] = {}
    for row in _load_jsonl(path):
        if "id" not in row or "text" not in row:
            raise ValueError(f"{path} rows must contain 'id' and 'text': {row}")
        record_id = str(row["id"])
        text = str(row["text"])
        if "title" in row and row["title"]:
            text = f"{row['title']} {text}"
        if not text.strip():
            text = "<empty>"
        if instruction:
            text = instruction + text
        records[record_id] = text
    return records


def _load_relevance(task_path: Path) -> Dict[str, Dict[str, int]]:
    jsonl_path = task_path / "relevance.jsonl"
    json_path = task_path / "relevance.json"

    relevance: Dict[str, Dict[str, int]] = {}
    if jsonl_path.exists():
        for row in _load_jsonl(jsonl_path):
            _merge_relevance(relevance, row, jsonl_path)
        return relevance

    if json_path.exists():
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Expected object in {json_path}")
        _merge_relevance(relevance, data, json_path)
        return relevance

    raise FileNotFoundError(
        f"Expected relevance.jsonl or relevance.json under {task_path}"
    )


def _merge_relevance(
    target: Dict[str, Dict[str, int]],
    source: Dict[str, Any],
    path: Path,
) -> None:
    for query_id, doc_scores in source.items():
        if not isinstance(doc_scores, dict):
            raise ValueError(f"Relevance for query {query_id!r} in {path} must be an object")
        target.setdefault(str(query_id), {})
        for doc_id, score in doc_scores.items():
            target[str(query_id)][str(doc_id)] = int(score)


def _load_custom_instructions(task_path: Path) -> Dict[str, str]:
    path = task_path / "custom_instruction.json"
    if not path.exists():
        return {
            "query": DEFAULT_QUERY_INSTRUCTION,
            "corpus": DEFAULT_CORPUS_INSTRUCTION,
        }

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected object in {path}")
    return {
        "query": str(data.get("query", DEFAULT_QUERY_INSTRUCTION)),
        "corpus": str(data.get("corpus", DEFAULT_CORPUS_INSTRUCTION)),
    }


def _validate_agentic_coding_dataset(dataset: AgenticCodingDataset) -> None:
    if not dataset.queries:
        raise ValueError(f"{dataset.name} has no queries")
    if not dataset.corpus:
        raise ValueError(f"{dataset.name} has no corpus documents")
    if not dataset.relevance:
        raise ValueError(f"{dataset.name} has no relevance judgments")

    missing_queries = set(dataset.relevance) - set(dataset.queries)
    if missing_queries:
        preview = sorted(missing_queries)[:5]
        raise ValueError(f"Relevance references unknown query ids: {preview}")

    missing_docs = {
        doc_id
        for doc_scores in dataset.relevance.values()
        for doc_id in doc_scores
        if doc_id not in dataset.corpus
    }
    if missing_docs:
        preview = sorted(missing_docs)[:5]
        raise ValueError(f"Relevance references unknown corpus ids: {preview}")
