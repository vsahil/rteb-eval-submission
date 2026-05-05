from rteb.agentic_coding_dataset import (
    AgenticCodingDataset,
    list_agentic_coding_tasks,
    load_agentic_coding_task,
)
from rteb.datasets import RTEBDataset, load_dataset
from rteb.metrics import compute_metrics

__version__ = "0.1.0"
__all__ = [
    "AgenticCodingDataset",
    "RTEBDataset",
    "compute_metrics",
    "list_agentic_coding_tasks",
    "load_agentic_coding_task",
    "load_dataset",
]


def __getattr__(name):
    if name == "RTEBEvaluator":
        from rteb.evaluator import RTEBEvaluator

        return RTEBEvaluator
    raise AttributeError(f"module 'rteb' has no attribute {name!r}")
