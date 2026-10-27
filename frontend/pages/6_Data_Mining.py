import streamlit as st
import pandas as pd
from utils.ui import load_css, show_tree
from backend.classification.models import run as run_classification
from backend.association_mining.rules import mine as mine_rules, compare_algorithms, prepare_transactions
from backend.clustering.kmeans import run as run_kmeans, elbow as run_elbow
from backend.visualization.charts import cluster_scatter, elbow_chart, confusion_heatmap, importance_bar

load_css()
st.title('6. Data Mining')

# Determine active dataset: prefer cleaned/preprocessed dataset, fallback to raw
df = st.session_state.get('processed_df')
is_preprocessed = df is not None
if df is None:
    df = st.session_state.get('raw_df')

if df is None:
    st.warning('No dataset loaded. Please go to **Page 1 (Dataset)** to upload or load a dataset first.')
    st.stop()

st.caption(f"Active dataset: **{'Cleaned / Preprocessed Data' if is_preprocessed else 'Raw Ingested Data'}** ({len(df)} rows, {len(df.columns)} columns)")

t1, t2, t3 = st.tabs(['Classification (Decision Tree & Others)', 'Association Rules (Apriori vs FP-Growth)', 'Clustering (K-Means)'])

# =========================================================================
# TAB 1: CLASSIFICATION
# =========================================================================
with t1:
    st.markdown("### Supervised Classification")
    all_cols = list(df.columns)

    c1, c2 = st.columns(2)
    with c1:
        target_default_idx = len(all_cols) - 1
        target = st.selectbox('Target Variable (y)', all_cols, index=target_default_idx, key='cls_target')
    with c2:
        alg = st.selectbox('Algorithm', ['Decision Tree', 'ID3', 'J48', 'Naïve Bayes', 'k-NN', 'Random Forest'], key='cls_alg')

    feature_options = [c for c in all_cols if c != target]
    features = st.multiselect('Input Features (X)', feature_options, default=feature_options, key='cls_features')

    # Configurable Decision Tree Hyperparameters
    crit = 'gini'
    max_d = None
    min_split = 2
    min_leaf = 1

    if 'tree' in alg.lower() or alg in ('ID3', 'J48'):
        with st.expander('Decision Tree Hyperparameters', expanded=True):
            p1, p2, p3, p4 = st.columns(4)
            with p1:
                if alg == 'Decision Tree':
                    crit = st.selectbox('Split Criterion', ['Gini Impurity', 'Entropy (Information Gain)'], index=0)
                    crit = 'entropy' if 'Entropy' in crit else 'gini'
                else:
                    st.caption(f"Criterion: **Entropy** ({alg} standard)")
                    crit = 'entropy'
            with p2:
                max_d_val = st.number_input('Max Tree Depth (0 = unlimited)', min_value=0, max_value=20, value=4)
                max_d = None if max_d_val == 0 else int(max_d_val)
            with p3:
                min_split = st.number_input('Min Samples Split', min_value=2, max_value=20, value=2)
            with p4:
                min_leaf = st.number_input('Min Samples Leaf', min_value=1, max_value=20, value=1)

    if st.button('Train & Evaluate Classifier', type='primary', key='btn_run_cls'):
        if not features:
            st.error("Please select at least one input feature.")
        else:
            with st.spinner(f"Training {alg}..."):
                try:
                    res = run_classification(
                        df, target=target, algorithm=alg, feature_cols=features,
                        criterion=crit, max_depth=max_d, min_samples_split=min_split,
                        min_samples_leaf=min_leaf
                    )
                    st.session_state.classification = res
                    st.success(f"{alg} trained and evaluated successfully!")
                except Exception as e:
                    st.session_state.pop('classification', None)
                    st.error(f"Classification error: {e}")

    r = st.session_state.get('classification')
    if r:
        m = r['metrics']
        st.divider()
        st.markdown(f"#### Results: {r['algorithm']} predicting `{r['target']}`")
        st.caption(f"Training set: {r['train_samples']} samples · Test set: {r['test_samples']} samples")

        # Metrics cards
        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
        col_m1.metric('Accuracy', f"{m['Accuracy'] * 100:.2f}%")
        col_m2.metric('Precision (Weighted)', f"{m['Precision']:.3f}")
        col_m3.metric('Recall (Weighted)', f"{m['Recall']:.3f}")
        col_m4.metric('F1-Score (Weighted)', f"{m['F1']:.3f}")
        col_m5.metric("Cohen's Kappa", f"{m['Kappa']:.3f}")

        st.dataframe(pd.DataFrame([{
            'Accuracy': round(m['Accuracy'], 4),
            'Precision (Weighted)': round(m['Precision'], 4),
            'Precision (Macro)': round(m['Precision (Macro)'], 4),
            'Recall (Weighted)': round(m['Recall'], 4),
            'Recall (Macro)': round(m['Recall (Macro)'], 4),
            'F1 (Weighted)': round(m['F1'], 4),
            'F1 (Macro)': round(m['F1 (Macro)'], 4),
            'Cohen Kappa': round(m['Kappa'], 4),
            'ROC-AUC': round(m['AUC'], 4) if m.get('AUC') is not None else 'N/A'
        }]), use_container_width=True)

        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader('Confusion Matrix')
            st.plotly_chart(confusion_heatmap(m['Confusion Matrix'], r['classes']), use_container_width=True)
        with col_right:
            st.subheader('Classification Report')
            report_df = pd.DataFrame(m['Report']).T
            st.dataframe(report_df, use_container_width=True)

        if r.get('importances') is not None and not r['importances'].empty:
            st.subheader('Feature Importance')
            st.plotly_chart(importance_bar(r['importances']), use_container_width=True)

        if r.get('tree_model') is not None:
            st.subheader('Decision Tree Architecture')
            show_tree(r, 'dm_tree')

# =========================================================================
# TAB 2: ASSOCIATION RULE MINING (APRIORI & FP-GROWTH)
# =========================================================================
with t2:
    st.markdown("### Association Rule Mining: Apriori vs FP-Growth")
    st.caption("Mine frequent itemsets and infer high-confidence, high-lift association rules.")

    format_choice = st.selectbox(
        'Transaction Data Format',
        ['Auto-detect', 'Basket string (e.g. comma-separated products)', 'Binary / One-Hot item matrix', 'Categorical tabular attributes'],
        key='arm_format'
    )
    format_map = {
        'Auto-detect': 'auto',
        'Basket string (e.g. comma-separated products)': 'basket',
        'Binary / One-Hot item matrix': 'binary',
        'Categorical tabular attributes': 'categorical'
    }

    c_alg, c_supp, c_conf = st.columns(3)
    with c_alg:
        arm_alg = st.selectbox('Algorithm', ['Apriori', 'FP-Growth'], key='arm_alg')
    with c_supp:
        min_supp = st.slider('Minimum Support (min_sup)', 0.01, 1.0, 0.10, 0.01, key='arm_supp')
    with c_conf:
        min_conf = st.slider('Minimum Confidence (min_conf)', 0.01, 1.0, 0.40, 0.01, key='arm_conf')

    btn_mine, btn_compare = st.columns(2)
    run_single = btn_mine.button(f'Mine with {arm_alg}', type='primary', key='btn_mine_rules')
    run_bench = btn_compare.button('⚖️ Compare Apriori vs FP-Growth Side-by-Side', key='btn_compare_arm')

    if run_single:
        with st.spinner(f"Running {arm_alg}..."):
            try:
                items, rules, tx, elapsed = mine_rules(
                    df, algorithm=arm_alg, support=min_supp, confidence=min_conf,
                    format_type=format_map[format_choice]
                )
                st.session_state.itemsets = items
                st.session_state.rules = rules
                st.session_state.arm_stats = {
                    'alg': arm_alg, 'elapsed_ms': round(elapsed * 1000, 2),
                    'n_tx': len(tx), 'n_items': len(tx.columns)
                }
                st.session_state.pop('arm_comparison', None)
                st.success(f"{arm_alg} completed in {round(elapsed * 1000, 2)} ms!")
            except Exception as e:
                st.error(f"Association Mining error: {e}")

    if run_bench:
        with st.spinner("Executing side-by-side benchmark on identical transactions..."):
            try:
                tx_matrix = prepare_transactions(df, format_type=format_map[format_choice])
                bench_res = compare_algorithms(tx_matrix, min_support=min_supp, min_confidence=min_conf)
                st.session_state.arm_comparison = bench_res
                st.session_state.itemsets = bench_res['items_fpgrowth']
                st.session_state.rules = bench_res['rules_fpgrowth']
                st.success("Comparison completed successfully!")
            except Exception as e:
                st.error(f"Comparison error: {e}")

    # Display comparison if available
    comp = st.session_state.get('arm_comparison')
    if comp:
        st.divider()
        st.subheader('Apriori vs FP-Growth Benchmark Comparison')
        st.markdown(f"**Dataset**: {comp['num_transactions']} transactions, {comp['num_items']} unique items · **Min Support**: {min_supp} · **Min Confidence**: {min_conf}")

        st.dataframe(comp['summary'], use_container_width=True)

        k1, k2, k3 = st.columns(3)
        k1.metric("Speedup Ratio", f"{comp['speedup']}x faster" if comp['speedup'] >= 1 else f"{round(1/comp['speedup'], 2)}x slower")
        k2.metric("Frequent Itemsets Consistency", "100% Identical" if comp['is_consistent'] else "Discrepancy Detected")
        k3.metric("Association Rules", f"{len(comp['rules_fpgrowth'])}")

        with st.expander('Theoretical Architecture Comparison (Apriori vs FP-Growth)'):
            st.markdown("""
            - **Candidate Generation**: Apriori uses the *Apriori property* to generate candidate itemsets of size $k$ ($C_k$) from frequent itemsets of size $k-1$ ($L_{k-1}$), scanning the database once per level $k$.
            - **FP-Tree Compression**: FP-Growth constructs a compact *Frequent Pattern Tree (FP-Tree)* in only **two scans** of the transaction database, completely bypassing candidate generation.
            - **Execution Pattern**: On dense datasets or low support thresholds where $C_k$ explodes exponentially, FP-Growth shows significant speedups.
            """)

    # Display single run results
    if 'itemsets' in st.session_state and not comp:
        stats = st.session_state.get('arm_stats', {})
        st.divider()
        st.subheader(f"Results for {stats.get('alg', 'Association Mining')}")
        st.caption(f"Processed {stats.get('n_tx')} transactions across {stats.get('n_items')} items in {stats.get('elapsed_ms')} ms")

    items = st.session_state.get('itemsets')
    rules = st.session_state.get('rules')

    if items is not None:
        c_i, c_r = st.columns([1, 1])
        with c_i:
            st.markdown(f"#### Frequent Itemsets ({len(items)} found)")
            if items.empty:
                st.info("No frequent itemsets found for the given minimum support.")
            else:
                display_items = items.copy()
                display_items['itemsets'] = display_items['itemsets'].apply(lambda x: ', '.join(sorted(list(x))))
                display_items['support'] = display_items['support'].round(4)
                st.dataframe(display_items, use_container_width=True)

        with c_r:
            st.markdown(f"#### Association Rules ({len(rules) if rules is not None else 0} generated)")
            if rules is None or rules.empty:
                st.info("No association rules found for the given minimum confidence.")
            else:
                display_rules = rules.copy()
                display_rules['antecedents'] = display_rules['antecedents'].apply(lambda x: ', '.join(sorted(list(x))))
                display_rules['consequents'] = display_rules['consequents'].apply(lambda x: ', '.join(sorted(list(x))))
                for col in ['support', 'confidence', 'lift', 'leverage']:
                    if col in display_rules.columns:
                        display_rules[col] = display_rules[col].round(4)
                st.dataframe(display_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']], use_container_width=True)

# =========================================================================
# TAB 3: CLUSTERING (K-MEANS)
# =========================================================================
with t3:
    st.markdown("### Unsupervised Clustering: K-Means")
    num_cols_available = df.select_dtypes(include=['number']).columns.tolist()

    if not num_cols_available:
        st.warning("No numeric columns detected in the current dataset. K-Means clustering requires numeric features.")
    else:
        col_feat, col_scale, col_k = st.columns([2, 1, 1])
        with col_feat:
            km_features = st.multiselect('Features for Clustering', num_cols_available, default=num_cols_available, key='km_features')
        with col_scale:
            km_scale_choice = st.selectbox('Feature Scaling', ['StandardScaler (Z-score)', 'MinMaxScaler ([0, 1])', 'None (Raw Distance)'], key='km_scale')
            scale_code = 'standard' if 'Standard' in km_scale_choice else ('minmax' if 'MinMax' in km_scale_choice else 'none')
        with col_k:
            k_val = st.slider('Number of Clusters (K)', min_value=2, max_value=min(10, max(2, len(df) - 1)), value=3, key='km_k')

        c_run_km, c_run_elbow = st.columns(2)
        btn_km = c_run_km.button('Run K-Means Clustering', type='primary', key='btn_run_km')
        btn_elbow = c_run_elbow.button('Run Elbow Method (Find Optimal K)', key='btn_run_elbow')

        if btn_km:
            if not km_features:
                st.error("Please select at least one numeric feature.")
            else:
                with st.spinner("Fitting K-Means model..."):
                    try:
                        res = run_kmeans(df, k=k_val, feature_cols=km_features, scale=scale_code)
                        st.session_state.kmeans_result = res
                        st.success(f"K-Means fitted successfully with K={k_val}!")
                    except Exception as e:
                        st.error(f"K-Means error: {e}")

        if btn_elbow:
            if not km_features:
                st.error("Please select at least one numeric feature.")
            else:
                with st.spinner("Evaluating Inertia and Silhouette across K=2..10..."):
                    try:
                        e_df, rec_k = run_elbow(df, kmax=10, feature_cols=km_features, scale=scale_code)
                        st.session_state.elbow_result = {'df': e_df, 'recommended_k': rec_k}
                        st.success(f"Elbow curve calculated. Recommended K: **{rec_k}**")
                    except Exception as e:
                        st.error(f"Elbow calculation error: {e}")

        # Display K-Means Results
        km_res = st.session_state.get('kmeans_result')
        if km_res:
            st.divider()
            st.subheader(f"K-Means Clustering Results (K={km_res['k']})")

            m_sse, m_sil = st.columns(2)
            m_sse.metric('Inertia / SSE (Sum of Squared Errors)', f"{km_res['inertia']:,.2f}")
            sil_val = km_res['silhouette']
            m_sil.metric('Silhouette Score', f"{sil_val:.4f}" if sil_val is not None else "N/A",
                         help="Range: -1 to +1. Values near +1 indicate dense, well-separated clusters.")

            col_cent, col_proj = st.columns([1, 1])
            with col_cent:
                st.markdown("#### Cluster Centroids (In Original Measurement Units)")
                st.caption("Values represent the mean feature values of each cluster in original dataset units:")
                st.dataframe(km_res['centroids'].round(3), use_container_width=True)

                st.markdown("#### Cluster Distribution")
                st.dataframe(km_res['profile'], use_container_width=True)

            with col_proj:
                st.markdown("#### 2D PCA Cluster Projection")
                ev = km_res['explained_variance']
                ev_str = f"PC1: {ev[0]*100:.1f}%, PC2: {ev[1]*100:.1f}%" if len(ev) >= 2 else ""
                st.caption(f"Variance captured: {ev_str}")
                st.plotly_chart(cluster_scatter(km_res['p'], km_res['labels']), use_container_width=True)

        # Display Elbow Method Results
        elbow_data = st.session_state.get('elbow_result')
        if elbow_data:
            st.divider()
            st.subheader(f"Elbow Analysis (Recommended K = {elbow_data['recommended_k']})")
            e_df = elbow_data['df']
            col_chart, col_tbl = st.columns([2, 1])
            with col_chart:
                st.plotly_chart(elbow_chart(e_df), use_container_width=True)
            with col_tbl:
                st.dataframe(e_df.round(3), use_container_width=True)
