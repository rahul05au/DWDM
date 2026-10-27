"""Data Preprocessing Module supporting Missing Value Imputation, Duplicate Handling,
Categorical Encoding, Scaling, and Outlier Detection.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

_NA_TOKENS = {'', 'na', 'n/a', 'nan', 'null', 'none', '?', '-', '--', 'missing', 'unknown'}


def infer_types(df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    """Convert text columns that really hold numbers to proper numeric types."""
    if df is None:
        return None
    x = df.copy()
    for c in x.columns:
        if pd.api.types.is_numeric_dtype(x[c]) or pd.api.types.is_bool_dtype(x[c]):
            continue
        s = x[c].astype('string').str.strip()
        s = s.mask(s.str.lower().isin(_NA_TOKENS))
        present = s.notna()
        if not present.any():
            continue
        t = (s.str.replace(r'[,\s]', '', regex=True)
              .str.replace(r'^[\$€£₹¥]', '', regex=True)
              .str.replace(r'%$', '', regex=True)
              .str.replace(r'^\((.*)\)$', r'-\1', regex=True))
        num = pd.to_numeric(t, errors='coerce')
        if num[present].notna().mean() >= threshold:
            x[c] = num.astype('float64')
            if not x[c].isna().any() and x[c].apply(float.is_integer).all():
                x[c] = x[c].astype('int64')
    return x


def profile_dataset(df: pd.DataFrame) -> dict:
    """Return diagnostic summary of missing values, duplicates, and column types."""
    if df is None or df.empty:
        return {'rows': 0, 'columns': 0, 'missing': 0, 'duplicates': 0, 'num_cols': [], 'cat_cols': []}

    df_clean = infer_types(df)
    num_cols = df_clean.select_dtypes(include=['number']).columns.tolist()
    cat_cols = [c for c in df_clean.columns if c not in num_cols]

    missing_series = df_clean.isna().sum()
    missing_cols = missing_series[missing_series > 0].to_dict()

    return {
        'rows': len(df_clean),
        'columns': len(df_clean.columns),
        'missing_total': int(missing_series.sum()),
        'missing_cols': missing_cols,
        'duplicates': int(df_clean.duplicated().sum()),
        'num_cols': num_cols,
        'cat_cols': cat_cols
    }


def detect_outliers_iqr(df: pd.DataFrame, numeric_cols=None, factor: float = 1.5):
    """Detect outlier indices in numeric columns using the Interquartile Range (IQR) rule."""
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

    outlier_summary = {}
    all_outlier_indices = set()

    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 4:
            continue
        q25, q75 = np.percentile(series, 25), np.percentile(series, 75)
        iqr = q75 - q25
        lower_bound = q25 - (factor * iqr)
        upper_bound = q75 + (factor * iqr)

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
        outlier_summary[col] = {
            'count': len(outliers),
            'lower_bound': round(float(lower_bound), 3),
            'upper_bound': round(float(upper_bound), 3)
        }
        all_outlier_indices.update(outliers)

    return list(all_outlier_indices), outlier_summary


def clean(df: pd.DataFrame,
          encode: bool = False,
          scale: str = 'None',
          remove_duplicates: bool = True,
          num_impute: str = 'median',
          cat_impute: str = 'mode',
          remove_outliers: bool = False,
          selected_columns=None) -> tuple[pd.DataFrame, dict]:
    """Execute complete data preprocessing pipeline.

    Args:
        df: Input raw DataFrame
        encode: Whether to one-hot encode categorical columns
        scale: 'None', 'StandardScaler', or 'MinMaxScaler'
        remove_duplicates: Whether to drop duplicate rows
        num_impute: 'median', 'mean', or 'zero'
        cat_impute: 'mode' or 'missing'
        remove_outliers: Whether to filter out IQR outliers
        selected_columns: Optional subset of columns to retain
    """
    if df is None:
        raise ValueError("No dataframe provided.")

    x = infer_types(df)
    if selected_columns:
        valid_cols = [c for c in selected_columns if c in x.columns]
        if valid_cols:
            x = x[valid_cols]

    rows_before = len(x)
    cols_before = len(x.columns)
    missing_before = int(x.isna().sum().sum())
    duplicates_before = int(x.duplicated().sum())

    operations_applied = []

    # 1. Duplicates
    duplicates_removed = 0
    if remove_duplicates and duplicates_before > 0:
        x = x.drop_duplicates()
        duplicates_removed = rows_before - len(x)
        operations_applied.append(f"Removed {duplicates_removed} duplicate rows")

    # 2. Missing values
    num_cols = x.select_dtypes(include=['number']).columns.tolist()
    cat_cols = [c for c in x.columns if c not in num_cols]

    imputed_count = 0
    for c in num_cols:
        n_missing = x[c].isna().sum()
        if n_missing > 0:
            imputed_count += n_missing
            if num_impute == 'mean':
                val = x[c].mean()
            elif num_impute == 'zero':
                val = 0
            else:
                val = x[c].median()
            x[c] = x[c].fillna(val)

    for c in cat_cols:
        n_missing = x[c].isna().sum()
        if n_missing > 0:
            imputed_count += n_missing
            if cat_impute == 'missing':
                x[c] = x[c].fillna('Missing')
            else:
                mode_val = x[c].mode()
                x[c] = x[c].fillna(mode_val.iloc[0] if len(mode_val) else 'Unknown')

    if imputed_count > 0:
        operations_applied.append(f"Imputed {imputed_count} missing values ({num_impute} numeric, {cat_impute} categorical)")

    # 3. Outlier removal
    outliers_removed = 0
    if remove_outliers and len(num_cols) > 0:
        outlier_indices, _ = detect_outliers_iqr(x, num_cols)
        if outlier_indices:
            outliers_removed = len(outlier_indices)
            x = x.drop(index=outlier_indices).reset_index(drop=True)
            operations_applied.append(f"Filtered {outliers_removed} outlier rows using 1.5x IQR rule")

    # 4. Encoding
    if encode and len(cat_cols) > 0:
        x = pd.get_dummies(x, columns=cat_cols, dtype=float)
        operations_applied.append(f"One-hot encoded {len(cat_cols)} categorical columns")

    # 5. Scaling
    if scale == 'StandardScaler' and len(num_cols) > 0:
        x[num_cols] = StandardScaler().fit_transform(x[num_cols])
        operations_applied.append("Standardized numeric columns with StandardScaler (mean=0, std=1)")
    elif scale == 'MinMaxScaler' and len(num_cols) > 0:
        x[num_cols] = MinMaxScaler().fit_transform(x[num_cols])
        operations_applied.append("Normalized numeric columns with MinMaxScaler (range [0, 1])")

    summary_info = {
        'rows_before': rows_before,
        'rows_after': len(x),
        'cols_before': cols_before,
        'cols_after': len(x.columns),
        'duplicates_removed': duplicates_removed,
        'missing_values_imputed': imputed_count,
        'outliers_removed': outliers_removed,
        'operations': operations_applied
    }

    return x, summary_info
