import os
from pathlib import Path
import streamlit as st
import pandas as pd
from utils.ui import load_css
from backend.data_input.data_loader import load_dataset, summary
from backend.preprocessing.preprocessing import profile_dataset

load_css()
st.title('1. Dataset Ingestion & Profiling')

SAMPLE_DIR = Path(__file__).resolve().parents[2] / 'sample_data'
SAMPLE_DATASETS = {
    'Select a sample dataset...': None,
    'Groceries Transactions (Market Basket / Association Mining)': SAMPLE_DIR / 'groceries_transactions.csv',
    'Mall Customers (Customer Segmentation / K-Means Clustering)': SAMPLE_DIR / 'mall_customers.csv',
    'Iris Flowers (Multiclass Classification & Clustering)': SAMPLE_DIR / 'iris_sample.csv'
}

def clear_downstream_state():
    """Clear downstream states to prevent stale results when a new dataset is loaded."""
    for key in ['processed_df', 'classification', 'kmeans', 'elbow', 'rules', 'itemsets', 'comparison']:
        st.session_state.pop(key, None)

st.subheader('Load or Upload Dataset')
c1, c2 = st.columns([1, 1])

with c1:
    chosen_sample = st.selectbox('Built-in Sample Datasets (for instant testing)', list(SAMPLE_DATASETS.keys()))
    if chosen_sample != 'Select a sample dataset...' and SAMPLE_DATASETS[chosen_sample]:
        path = SAMPLE_DATASETS[chosen_sample]
        if path.exists():
            if st.session_state.get('last_sample') != chosen_sample:
                clear_downstream_state()
                df = pd.read_csv(path)
                st.session_state.raw_df = df
                st.session_state.dataset_name = path.name
                st.session_state.last_sample = chosen_sample
                st.success(f"Loaded '{chosen_sample}' successfully!")

with c2:
    f = st.file_uploader('Or Upload Custom File (CSV, XLS/XLSX, ARFF, ZIP)', type=['csv', 'xls', 'xlsx', 'arff', 'zip'])
    if f is not None:
        if st.session_state.get('uploaded_file_name') != f.name:
            try:
                clear_downstream_state()
                df = load_dataset(f)
                st.session_state.raw_df = df
                st.session_state.dataset_name = f.name
                st.session_state.uploaded_file_name = f.name
                st.session_state.last_sample = None
                st.success(f"Uploaded and parsed '{f.name}' successfully!")
            except Exception as e:
                st.error(f"Error reading file: {e}")

df = st.session_state.get('raw_df')
if df is not None:
    st.divider()
    prof = profile_dataset(df)
    st.subheader(f"Dataset Overview: {st.session_state.get('dataset_name', 'Current Dataset')}")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Rows", f"{prof['rows']:,}")
    m2.metric("Total Columns", f"{prof['columns']}")
    m3.metric("Missing Values", f"{prof['missing_total']:,}")
    m4.metric("Duplicate Rows", f"{prof['duplicates']:,}")

    t1, t2, t3 = st.tabs(['Data Preview', 'Column & Types Analysis', 'Descriptive Statistics'])

    with t1:
        st.dataframe(df.head(50), use_container_width=True)

    with t2:
        dtypes_df = pd.DataFrame({
            'Column': df.columns,
            'Data Type': [str(t) for t in df.dtypes],
            'Non-Null Count': df.notna().sum().values,
            'Null Count': df.isna().sum().values,
            'Null %': (df.isna().mean() * 100).round(2).values,
            'Unique Values': df.nunique().values
        })
        st.dataframe(dtypes_df, use_container_width=True)

    with t3:
        st.dataframe(df.describe(include='all').T, use_container_width=True)

    st.info("Dataset is ingested and held in session state. Proceed to **Page 2 (Data Preprocessing)** or directly to **Page 6 (Data Mining)**.")
else:
    st.info("Upload a file or pick one of the built-in sample datasets above to begin.")
