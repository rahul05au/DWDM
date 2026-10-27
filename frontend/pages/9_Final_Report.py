import io
import json
import streamlit as st
import pandas as pd
from utils.ui import load_css
from backend.final_report.report import pdf_bytes, html_bytes, excel_bytes

load_css()
st.title('9. Final Report & Export')

df = st.session_state.get('processed_df', st.session_state.get('raw_df'))
if df is None:
    st.warning('No dataset available. Please ingest data on Page 1 first.')
    st.stop()

cls = st.session_state.get('classification')
km = st.session_state.get('kmeans_result')
rules = st.session_state.get('rules')
items = st.session_state.get('itemsets')
comp = st.session_state.get('arm_comparison')

sections = []

# 1. Dataset Section
ds_text = f"Dataset: {st.session_state.get('dataset_name', 'Loaded Dataset')}\n" \
          f"Rows: {len(df)}\nColumns: {len(df.columns)}\n" \
          f"Missing Values: {int(df.isna().sum().sum())}\n" \
          f"Duplicates: {int(df.duplicated().sum())}\n" \
          f"Columns: {', '.join(list(df.columns))}"
sections.append(('1. Dataset Overview', ds_text))

# 2. Classification Section
excel_sheets = {'Dataset': df.head(1000)}

if cls:
    m = cls['metrics']
    cls_summary = {
        'Algorithm': cls['algorithm'],
        'Target': cls['target'],
        'Accuracy': round(m['Accuracy'], 4),
        'Precision (Weighted)': round(m['Precision'], 4),
        'Precision (Macro)': round(m['Precision (Macro)'], 4),
        'Recall (Weighted)': round(m['Recall'], 4),
        'Recall (Macro)': round(m['Recall (Macro)'], 4),
        'F1-Score (Weighted)': round(m['F1'], 4),
        'F1-Score (Macro)': round(m['F1 (Macro)'], 4),
        'Cohen Kappa': round(m['Kappa'], 4),
        'ROC-AUC': round(m['AUC'], 4) if m.get('AUC') is not None else 'N/A'
    }
    sections.append(('2. Classification Results', json.dumps(cls_summary, indent=2)))
    excel_sheets['Classification_Metrics'] = pd.DataFrame([cls_summary])
    excel_sheets['Confusion_Matrix'] = pd.DataFrame(m['Confusion Matrix'], index=cls['classes'], columns=cls['classes'])

# 3. Clustering Section
if km:
    km_summary = {
        'Algorithm': 'K-Means',
        'K (Clusters)': km['k'],
        'Inertia / SSE': round(km['inertia'], 2),
        'Silhouette Score': round(km['silhouette'], 4) if km['silhouette'] is not None else 'N/A',
        'Scaling Used': km['scale_used']
    }
    sections.append(('3. Clustering Results', json.dumps(km_summary, indent=2)))
    excel_sheets['Clustering_Centroids'] = km['centroids']
    excel_sheets['Cluster_Distribution'] = km['profile']

# 4. Association Rules Section
if rules is not None and not rules.empty:
    arm_summary = {
        'Frequent Itemsets Count': len(items) if items is not None else 0,
        'Association Rules Count': len(rules),
        'Mean Support': round(rules['support'].mean(), 4),
        'Mean Confidence': round(rules['confidence'].mean(), 4),
        'Mean Lift': round(rules['lift'].mean(), 4)
    }
    sections.append(('4. Association Rule Mining', json.dumps(arm_summary, indent=2)))
    disp_r = rules.copy()
    disp_r['antecedents'] = disp_r['antecedents'].apply(lambda s: ', '.join(sorted(list(s))))
    disp_r['consequents'] = disp_r['consequents'].apply(lambda s: ', '.join(sorted(list(s))))
    excel_sheets['Association_Rules'] = disp_r[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(500)

if comp:
    sections.append(('5. Apriori vs FP-Growth Benchmark', comp['summary'].to_string(index=False)))
    excel_sheets['ARM_Benchmark'] = comp['summary']

st.subheader('Generated Academic Capstone Report')
st.markdown("Download comprehensive reports containing the complete lifecycle: raw dataset statistics, data preprocessing, model execution, evaluation metrics, and comparison analyses.")

d1, d2, d3 = st.columns(3)
with d1:
    st.download_button(
        label='📄 Download PDF Report',
        data=pdf_bytes('Data Mining Capstone Project Report', sections),
        file_name='DWDM_Capstone_Report.pdf',
        mime='application/pdf',
        use_container_width=True
    )

with d2:
    st.download_button(
        label='🌐 Download HTML Report',
        data=html_bytes('Data Mining Capstone Project Report', sections),
        file_name='DWDM_Capstone_Report.html',
        mime='text/html',
        use_container_width=True
    )

with d3:
    st.download_button(
        label='📊 Download Multi-Sheet Excel',
        data=excel_bytes(excel_sheets),
        file_name='DWDM_Capstone_Results.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        use_container_width=True
    )

st.divider()
st.subheader('Dataset Exports')
e1, e2 = st.columns(2)
with e1:
    st.download_button(
        label='Export Active Dataset as CSV',
        data=df.to_csv(index=False).encode('utf-8'),
        file_name='active_dataset.csv',
        mime='text/csv',
        use_container_width=True
    )
with e2:
    st.download_button(
        label='Export Active Dataset as JSON',
        data=df.to_json(orient='records', indent=2).encode('utf-8'),
        file_name='active_dataset.json',
        mime='application/json',
        use_container_width=True
    )
