from pathlib import Path

from rteb.agentic_coding_dataset import (
    list_agentic_coding_tasks,
    load_agentic_coding_task,
)


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "agentic_coding_toy"


def test_list_agentic_coding_tasks():
    assert list_agentic_coding_tasks(FIXTURE_ROOT) == ["toy_task"]


def test_load_agentic_coding_task():
    dataset = load_agentic_coding_task(FIXTURE_ROOT, "toy_task")

    assert dataset.name == "toy_task"
    assert dataset.num_queries == 1
    assert dataset.num_documents == 2
    assert dataset.queries["issue-1"] == "Fix the login redirect bug after session timeout."
    assert dataset.corpus["auth/session.py"] == "Session timeout and redirect handling code."
    assert dataset.relevance == {"issue-1": {"auth/session.py": 1}}


def test_load_agentic_coding_task_with_instructions():
    dataset = load_agentic_coding_task(
        FIXTURE_ROOT,
        "toy_task",
        add_instructions=True,
    )

    assert dataset.queries["issue-1"].startswith("Represent the query")
    assert dataset.corpus["auth/session.py"].startswith("Represent the document")
