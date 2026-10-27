import pytest
import pandas as pd
import numpy as np
from backend.classification.models import run


@pytest.fixture
def sample_classification_data():
    np.random.seed(42)
    n = 60
    return pd.DataFrame({
        'feature1': np.random.normal(loc=0, scale=1, size=n),
        'feature2': np.random.normal(loc=5, scale=2, size=n),
        'category': np.random.choice(['A', 'B', 'C'], size=n),
        'target': np.random.choice(['class1', 'class2'], size=n)
    })


def test_decision_tree_gini(sample_classification_data):
    res = run(sample_classification_data, target='target', algorithm='Decision Tree', criterion='gini', max_depth=3)
    assert res is not None
    assert 'metrics' in res
    assert 0.0 <= res['metrics']['Accuracy'] <= 1.0
    assert 0.0 <= res['metrics']['F1'] <= 1.0
    assert res['metrics']['Confusion Matrix'] is not None
    assert res['tree_model'] is not None


def test_decision_tree_entropy(sample_classification_data):
    res = run(sample_classification_data, target='target', algorithm='Decision Tree', criterion='entropy', max_depth=2)
    assert res is not None
    assert 'Accuracy' in res['metrics']
    assert res['tree'] is not None


def test_id3_and_j48(sample_classification_data):
    res_id3 = run(sample_classification_data, target='target', algorithm='ID3')
    res_j48 = run(sample_classification_data, target='target', algorithm='J48')

    assert res_id3['metrics']['Accuracy'] is not None
    assert res_j48['metrics']['Accuracy'] is not None
    assert res_id3['tree_model'] is not None
    assert res_j48['tree_model'] is not None


def test_invalid_target_raises_error(sample_classification_data):
    with pytest.raises(ValueError):
        run(sample_classification_data, target='non_existent_column', algorithm='Decision Tree')
