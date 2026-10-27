import pytest
import pandas as pd
from backend.association_mining.rules import prepare_transactions, mine, compare_algorithms


@pytest.fixture
def sample_basket_data():
    return pd.DataFrame({
        'TransactionID': [1, 2, 3, 4, 5, 6, 7],
        'Items': [
            'milk, bread, butter',
            'bread, butter',
            'milk, bread',
            'milk, cookies',
            'bread, butter, cookies',
            'milk, bread, butter',
            'bread, butter'
        ]
    })


def test_prepare_transactions_basket(sample_basket_data):
    tx = prepare_transactions(sample_basket_data, format_type='basket', item_col='Items')
    assert len(tx) == 7
    assert 'milk' in tx.columns
    assert 'bread' in tx.columns
    assert 'butter' in tx.columns
    assert tx.dtypes['milk'] == bool


def test_apriori_mining(sample_basket_data):
    items, rules, tx, elapsed = mine(sample_basket_data, algorithm='Apriori', support=0.2, confidence=0.5, format_type='basket')
    assert not items.empty
    assert 'support' in items.columns
    assert 'itemsets' in items.columns
    assert elapsed > 0
    # Bread and butter should be a frequent itemset
    assert any({'bread', 'butter'}.issubset(s) for s in items['itemsets'])


def test_fpgrowth_mining(sample_basket_data):
    items, rules, tx, elapsed = mine(sample_basket_data, algorithm='FP-Growth', support=0.2, confidence=0.5, format_type='basket')
    assert not items.empty
    assert 'support' in items.columns
    assert elapsed > 0


def test_empty_results_handling(sample_basket_data):
    # Support 1.0 (impossible high) should return empty dataframe gracefully without crash
    items, rules, tx, elapsed = mine(sample_basket_data, algorithm='Apriori', support=0.99, confidence=0.99, format_type='basket')
    assert items.empty
    assert rules.empty
