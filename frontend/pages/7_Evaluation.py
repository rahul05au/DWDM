import streamlit as st
import pandas as pd
from utils.ui import load_css
from backend.visualization.charts import confusion_heatmap, metrics_bar

load_css()
st.title('7. Evaluation Dashboard')

cls = st.session_state.get('classification')
km = st.session_state.get('kmeans_result')
rules = st.session_state.get('rules')
comp = st.session_state.get('arm_comparison')

if not any([cls, km, rules is not None and not rules.empty, comp]):
    st.info('No model evaluation results available yet. Please execute algorithms on **Page 6 (Data Mining)** first.')
    st.stop()

t1, t2, t3 = st.tabs(['Classification Evaluation', 'Clustering Evaluation', 'Association Mining Evaluation'])

# =========================================================================
# TAB 1: CLASSIFICATION EVALUATION
# =========================================================================
with t1:
    if not cls:
        st.info("No classification model has been evaluated yet. Run Decision Tree or another classifier on Page 6.")
    else:
        st.subheader(f"Classification Model: {cls['algorithm']} on `{cls['target']}`")
        m = cls['metrics']

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{m['Accuracy'] * 100:.2f}%")
        c2.metric("Weighted F1", f"{m['F1']:.4f}")
        c3.metric("Macro F1", f"{m['F1 (Macro)']:.4f}")
        c4.metric("Cohen's Kappa", f"{m['Kappa']:.4f}")

        st.plotly_chart(metrics_bar(m), use_container_width=True)

        col_cm, col_cr = st.columns(2)
        with col_cm:
            st.markdown("#### Confusion Matrix Heatmap")
            st.plotly_chart(confusion_heatmap(m['Confusion Matrix'], cls['classes']), use_container_width=True)
        with col_cr:
            st.markdown("#### Detailed Classification Report")
            st.dataframe(pd.DataFrame(m['Report']).T.round(3), use_container_width=True)

# =========================================================================
# TAB 2: CLUSTERING EVALUATION
# =========================================================================
with t2:
    if not km:
        st.info("No clustering model has been evaluated yet. Run K-Means on Page 6.")
    else:
        st.subheader(f"K-Means Clustering Evaluation (K={km['k']})")
        m_sse, m_sil, m_scale = st.columns(3)
        m_sse.metric("Inertia / SSE", f"{km['inertia']:,.2f}")
        sil = km['silhouette']
        m_sil.metric("Silhouette Score", f"{sil:.4f}" if sil is not None else "N/A")
        m_scale.metric("Distance Scaling Applied", km['scale_used'].title())

        st.markdown("#### Centroids (Original Measurement Units)")
        st.dataframe(km['centroids'].round(3), use_container_width=True)

        st.markdown("#### Cluster Sample Distribution")
        st.dataframe(km['profile'], use_container_width=True)

# =========================================================================
# TAB 3: ASSOCIATION MINING EVALUATION
# =========================================================================
with t3:
    if rules is None or rules.empty:
        st.info("No association rules have been mined yet. Run Apriori or FP-Growth on Page 6.")
    else:
        st.subheader("Association Rules Quality Metrics")
        m_supp, m_conf, m_lift = st.columns(3)
        m_supp.metric("Mean Support", f"{rules['support'].mean():.4f}")
        m_conf.metric("Mean Confidence", f"{rules['confidence'].mean():.4f}")
        m_lift.metric("Mean Lift", f"{rules['lift'].mean():.4f}")

        display_rules = rules.copy()
        display_rules['antecedents'] = display_rules['antecedents'].apply(lambda x: ', '.join(sorted(list(x))))
        display_rules['consequents'] = display_rules['consequents'].apply(lambda x: ', '.join(sorted(list(x))))
        st.dataframe(display_rules.round(4), use_container_width=True)

        if comp:
            st.markdown("#### Apriori vs FP-Growth Benchmark Summary")
            st.dataframe(comp['summary'], use_container_width=True)
