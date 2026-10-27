import pytest
import pandas as pd
import numpy as np
from backend.preprocessing.preprocessing import infer_types, clean, detect_outliers_iqr, profile_dataset


def test_infer_types():
    df = pd.DataFrame({
        'num_str': ['100', '200', ' 300 ', '$400', '500%'],
        'real_text': ['apple', 'banana', 'orange', 'grape', 'mango']
    })
    inferred = infer_types(df)
    assert pd.api.types.is_numeric_dtype(inferred['num_str'])
    assert not pd.api.types.is_numeric_dtype(inferred['real_text'])


def test_profile_dataset():
    df = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 1],
        'B': ['x', 'y', 'z', 'w', 'x']
    })
    prof = profile_dataset(df)
    assert prof['rows'] == 5
    assert prof['columns'] == 2
    assert prof['missing_total'] == 1
    assert prof['duplicates'] == 1


def test_clean_imputation_and_duplicates():
    df = pd.DataFrame({
        'A': [10.0, 20.0, np.nan, 40.0, 10.0],
        'B': ['cat', 'dog', np.nan, 'dog', 'cat']
    })
    cleaned, info = clean(df, remove_duplicates=True, num_impute='median', cat_impute='mode')
    # Row 4 is duplicate of row 0
    assert len(cleaned) == 4
    assert info['duplicates_removed'] == 1
    # Check no missing values remain
    assert cleaned.isna().sum().sum() == 0
    # Median of [10, 20, 40] is 20
    assert 20.0 in cleaned['A'].values


def test_outlier_detection_iqr():
    df = pd.DataFrame({
        'val': [10, 11, 12, 11, 10, 12, 11, 10, 1000]  # 1000 is obvious outlier
    })
    indices, summary = detect_outliers_iqr(df, ['val'], factor=1.5)
    assert 8 in indices
    assert summary['val']['count'] == 1

    cleaned, info = clean(df, remove_duplicates=False, remove_outliers=True)
    assert len(cleaned) == 8
    assert info['outliers_removed'] == 1
