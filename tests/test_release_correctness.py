import json
import sys

import pytest

from rteb.datasets import RTEBDataset
from rteb.evaluator import RTEBEvaluator
from rteb.metrics import compute_metrics
from rteb.models import BM25Model


def test_missing_predictions_count_as_zero():
    result = compute_metrics(
        {'q1': ['d1']}, {'q1': {'d1': 1}, 'q2': {'d2': 1}}, k_values=[1]
    )
    assert result == {'ndcg_at_1': 0.5, 'recall_at_1': 0.5}


def test_failed_dataset_aborts_instead_of_reporting_partial_average(monkeypatch):
    def fail(*args, **kwargs):
        raise OSError('dataset unavailable')
    monkeypatch.setattr('rteb.evaluator.load_dataset', fail)
    with pytest.raises(RuntimeError, match='incomplete benchmark'):
        RTEBEvaluator(datasets=['AILA_casedocs']).evaluate(BM25Model())


def test_empty_adapter_output_is_rejected(monkeypatch):
    monkeypatch.setattr(
        'rteb.evaluator.load_dataset',
        lambda *args, **kwargs: RTEBDataset('empty', 'legal', 'english'),
    )
    with pytest.raises(ValueError, match='empty queries, corpus, or qrels'):
        RTEBEvaluator(datasets=['empty']).evaluate(BM25Model())


def test_save_results_accepts_filename_without_parent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    RTEBEvaluator(datasets=[]).save_results({}, 'results.json')
    assert json.loads((tmp_path / 'results.json').read_text())['num_datasets'] == 0


@pytest.mark.parametrize('flag', ['--list-datasets', '--list-domains'])
def test_cli_can_list_without_loading_model(flag, monkeypatch, capsys):
    from rteb.evaluate import main
    monkeypatch.setattr(sys, 'argv', ['rteb-evaluate', flag])
    main()
    assert 'legal' in capsys.readouterr().out.lower()
