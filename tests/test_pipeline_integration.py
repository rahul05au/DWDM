import pytest
from pathlib import Path
import pandas as pd
from backend.preprocessing.preprocessing import clean, profile_dataset
from backend.classification.models import run as run_cls
from backend.clustering.kmeans import run as run_km
from backend.association_mining.rules import mine as run_arm, compare_algorithms, prepare_transactions
from backend.final_report.report import pdf_bytes, html_bytes, excel_bytes


def test_full_classification_pipeline():
    # 1. Ingest
    iris_path = Path(__file__).resolve().parents[1] / 'sample_data' / 'iris_sample.csv'
    raw_df = pd.read_csv(iris_path)
    prof = profile_dataset(raw_df)
    assert prof['rows'] == 150

    # 2. Preprocess
    cleaned_df, info = clean(raw_df, remove_duplicates=True, num_impute='median')
    assert len(cleaned_df) <= 150

    # 3. Model & Evaluate
    res = run_cls(cleaned_df, target='species', algorithm='Decision Tree', criterion='entropy', max_depth=3)
    assert res['metrics']['Accuracy'] > 0.80

    # 4. Report
    pdf = pdf_bytes('Test Report', [('Classification', str(res['metrics']))])
    assert len(pdf) > 0


def test_full_clustering_pipeline():
    # 1. Ingest Mall Customers
    mall_path = Path(__file__).resolve().parents[1] / 'sample_data' / 'mall_customers.csv'
    raw_df = pd.read_csv(mall_path)

    # 2. Preprocess
    cleaned_df, _ = clean(raw_df, remove_duplicates=True)

    # 3. K-Means
    km_res = run_km(cleaned_df, k=5, feature_cols=['Annual_Income_k', 'Spending_Score'], scale='standard')
    assert km_res['k'] == 5
    assert km_res['centroids'].shape == (5, 2)
    assert km_res['silhouette'] is not None


def test_full_association_pipeline():
    # 1. Ingest Groceries Transactions
    groc_path = Path(__file__).resolve().parents[1] / 'sample_data' / 'groceries_transactions.csv'
    raw_df = pd.read_csv(groc_path)

    # 2. Mine & Compare
    tx = prepare_transactions(raw_df, format_type='basket', item_col='Items')
    comp = compare_algorithms(tx, min_support=0.1, min_confidence=0.3)
    assert comp['is_consistent'] is True
    assert len(comp['items_fpgrowth']) > 0

    # 3. Export to Excel
    excel = excel_bytes({'Rules': comp['rules_fpgrowth'], 'Summary': comp['summary']})
    assert len(excel) > 0


def test_etl_and_database_warehouse():
    from backend.etl.etl import run as run_etl
    from backend.database.database import get_engine, test_connection, read_table, table_row_count

    # Test database connectivity
    assert test_connection() is True

    # Test ETL execution
    sample_df = pd.DataFrame({
        'Department': ['Sales', 'Engineering', 'Sales', 'HR'],
        'Salary': [50000, 80000, 55000, 60000]
    })
    ex, tr, info = run_etl(sample_df, table="test_emp_warehouse", dataset_name="test_employees")
    assert len(tr) == 4
    assert info['table'] == "test_emp_warehouse"

    # Verify table can be queried from SQLite
    count = table_row_count("test_emp_warehouse")
    assert count == 4
    loaded_df = read_table("test_emp_warehouse")
    assert len(loaded_df) == 4
    assert 'Department' in loaded_df.columns


def test_olap_operations():
    from backend.olap.olap import slice_df, dice, rollup, pivot

    df = pd.DataFrame({
        'Region': ['East', 'East', 'West', 'West'],
        'Category': ['Electronics', 'Furniture', 'Electronics', 'Furniture'],
        'Revenue': [1000, 400, 1500, 600]
    })

    # 1. Slice
    sliced = slice_df(df, 'Region', 'East')
    assert len(sliced) == 2
    assert (sliced['Region'] == 'East').all()

    # 2. Dice
    diced = dice(df, {'Region': ['East', 'West'], 'Category': ['Electronics']})
    assert len(diced) == 2
    assert (diced['Category'] == 'Electronics').all()

    # 3. Rollup
    rolled = rollup(df, ['Region', 'Category'], 'Revenue', 'sum')
    assert 'level' in rolled.columns
    assert len(rolled) > 0

    # 4. Pivot
    pivoted = pivot(df, index='Region', columns='Category', values='Revenue', agg='sum')
    assert 'Electronics' in pivoted.columns
    assert 'Furniture' in pivoted.columns

