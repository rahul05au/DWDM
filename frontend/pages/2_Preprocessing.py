import streamlit as st
import pandas as pd
from utils.ui import load_css
from backend.preprocessing.preprocessing import clean, profile_dataset

load_css()
st.title('2. Data Preprocessing')

df = st.session_state.get('raw_df')
if df is None:
    st.warning('Please upload or load a dataset first on Page 1 (Dataset).')
    st.stop()

st.caption(f"Currently active dataset: **{st.session_state.get('dataset_name', 'Loaded Data')}** ({len(df)} rows, {len(df.columns)} columns)")

with st.expander('Configure Preprocessing Options', expanded=True):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Cleaning & Imputation")
        dup = st.checkbox('Remove duplicate rows', value=True)
        num_impute = st.selectbox('Numeric missing-value imputation', ['median', 'mean', 'zero'], index=0)
        cat_impute = st.selectbox('Categorical missing-value imputation', ['mode', 'missing'], index=0)
        outliers = st.checkbox('Detect & filter outliers (1.5× IQR rule)', value=False, help="Removes extreme rows based on the Interquartile Range method.")

    with col2:
        st.markdown("#### Transformations & Encoding")
        enc = st.checkbox('One-hot encode categorical features', value=False, help="Converts text columns into binary dummy indicators. Note: keep unchecked if running Decision Tree or text category charts.")
        scale = st.selectbox('Feature Scaling', ['None', 'StandardScaler (Z-score)', 'MinMaxScaler ([0, 1])'], index=0)
        scale_val = 'StandardScaler' if 'Standard' in scale else ('MinMaxScaler' if 'MinMax' in scale else 'None')

        all_cols = list(df.columns)
        sel_cols = st.multiselect('Feature selection (columns to keep)', all_cols, default=all_cols)

if st.button('Run Preprocessing Pipeline', type='primary'):
    try:
        cleaned_df, info = clean(
            df,
            encode=enc,
            scale=scale_val,
            remove_duplicates=dup,
            num_impute=num_impute,
            cat_impute=cat_impute,
            remove_outliers=outliers,
            selected_columns=sel_cols
        )
        st.session_state.processed_df = cleaned_df
        st.session_state.preprocessing_info = info
        st.success("Preprocessing completed successfully!")
    except Exception as e:
        st.error(f"Preprocessing error: {e}")

cleaned_df = st.session_state.get('processed_df')
if cleaned_df is not None:
    st.divider()
    st.subheader('Preprocessing Results & Verification')

    info = st.session_state.get('preprocessing_info', {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows (Before → After)", f"{info.get('rows_before', len(df))} → {len(cleaned_df)}")
    c2.metric("Columns", f"{info.get('cols_before', len(df.columns))} → {len(cleaned_df.columns)}")
    c3.metric("Duplicates Removed", f"{info.get('duplicates_removed', 0)}")
    c4.metric("Outliers Filtered", f"{info.get('outliers_removed', 0)}")

    if info.get('operations'):
        with st.expander('Operations Log'):
            for op in info['operations']:
                st.write(f"- {op}")

    t1, t2 = st.tabs(['Cleaned Data Preview', 'Before vs After Statistics'])
    with t1:
        st.dataframe(cleaned_df.head(25), use_container_width=True)
        csv_data = cleaned_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Cleaned CSV",
            data=csv_data,
            file_name=f"cleaned_{st.session_state.get('dataset_name', 'data.csv')}",
            mime="text/csv"
        )

    with t2:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Original Dataset Summary**")
            st.dataframe(df.describe(include='all').T, use_container_width=True)
        with col_b:
            st.markdown("**Cleaned Dataset Summary**")
            st.dataframe(cleaned_df.describe(include='all').T, use_container_width=True)

    st.info("The cleaned dataset is now active in `st.session_state.processed_df` and will be used by ETL and downstream Data Mining algorithms.")
