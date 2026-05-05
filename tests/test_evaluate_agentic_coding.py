import json
from pathlib import Path

from rteb.evaluate_agentic_coding import (
    LexicalOverlapModel,
    evaluate_agentic_coding_tasks,
)


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "agentic_coding_toy"


def test_evaluate_agentic_coding_tasks(tmp_path):
    results = evaluate_agentic_coding_tasks(
        data_path=FIXTURE_ROOT,
        task_names=["toy_task"],
        model=LexicalOverlapModel(),
        output_path=tmp_path,
        show_progress=False,
    )

    assert "toy_task" in results
    assert results["toy_task"]["recall_at_1"] == 1.0
    assert results["toy_task"]["ndcg_at_1"] == 1.0

    eval_path = tmp_path / "toy_task" / "retrieve_eval.json"
    pred_path = tmp_path / "toy_task" / "retrieve_pred.json"
    summary_path = tmp_path / "summary.json"

    assert eval_path.exists()
    assert pred_path.exists()
    assert summary_path.exists()

    saved_eval = json.loads(eval_path.read_text())
    assert saved_eval["recall_at_1"] == 1.0
