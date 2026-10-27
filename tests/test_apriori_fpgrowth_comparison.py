import pytest
import pandas as pd
from backend.association_mining.rules import prepare_transactions, compare_algorithms


def test_apriori_vs_fpgrowth_consistency_and_benchmark():
    # Use grocery transactions
    df = pd.DataFrame({
        'items': [
            'milk, bread, butter',
            'bread, butter',
            'milk, bread',
            'milk, cookies, coffee',
            'bread, butter, cookies',
            'milk, bread, butter',
            'bread, butter',
            'milk, bread, butter, cookies',
            'coffee, cookies',
            'bread, tea, milk'
        ]
    })
    tx = prepare_transactions(df, format_type='basket', item_col='items')
    comp = compare_algorithms(tx, min_support=0.2, min_confidence=0.5)

    assert comp['is_consistent'] is True, "Apriori and FP-Growth MUST produce identical frequent itemsets under identical support settings"
    assert len(comp['items_apriori']) == len(comp['items_fpgrowth'])
    assert comp['summary']['Execution Time (ms)'].iloc[0] >= 0
    assert comp['summary']['Execution Time (ms)'].iloc[1] >= 0
    assert comp['speedup'] > 0
