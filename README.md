# Integrated Data Warehousing & Data Mining (DWDM) Visualization System

A comprehensive fourth-year B.Tech CSE Capstone Project implementing the full Data Mining lifecycle: from raw multi-format ingestion and preprocessing to algorithm execution (Decision Trees, K-Means, Apriori, FP-Growth), evaluation, visual analytics, and multi-format reporting.

---

## Architecture & End-to-End Pipeline

```
Raw Dataset (CSV / Excel / ARFF / ZIP)
  │
  ▼
[Page 1: Ingestion & Profiling] ─── Shape, Data Types, Missing & Duplicate Summary
  │
  ▼
[Page 2: Preprocessing Pipeline] ── Median/Mean Imputation, 1.5x IQR Outlier Filtering,
  │                                 MinMax/Standard Scaling, Categorical One-Hot Encoding
  ▼
Cleaned Dataset (st.session_state.processed_df)
  ├──► [Pages 3 & 4: ETL & SQLite Warehouse] ── Transactional Storage (dwdm_warehouse.db)
  │                                              and Schema Auditing
  ├──► [Page 5: OLAP Multidimensional Analysis] ─ Roll-up, Drill-down, Slice, Dice, Pivot
  │
  ▼
[Page 6: Data Mining Algorithms]
  ├── Classification: Decision Tree (Gini/Entropy, Max Depth, Stratified Split), ID3, J48, Naive Bayes, k-NN, Random Forest
  ├── Clustering: K-Means (Distance Scaling, Centroids in Original Units, Silhouette, Elbow Curve)
  └── Association Rules: Apriori vs FP-Growth (Multi-format parser, Side-by-Side High-Res Benchmark)
  │
  ▼
[Page 7: Unified Evaluation Dashboard] ──── Confusion Matrix, Accuracy/F1/AUC, Silhouette/Inertia, Support/Confidence/Lift
  │
  ▼
[Page 8: Visual Analytics] ─────────────── Interactive Plotly Decision Trees, PCA Clusters, Correlation Heatmaps
  │
  ▼
[Page 9: Multi-Format Report Export] ───── Downloadable PDF, HTML, Multi-Sheet Excel, CSV, JSON
  │
  ▼
[Page 10: Academic Guide & Viva Voce] ──── Provisional Syllabus Coverage Matrix, Viva Q&A, Demonstration Script
```

---

## 4 Core Algorithms — Implementation Highlights

### 1. Decision Tree Classification (Supervised)
- **Mathematical Criteria**: User-selectable **Entropy (Information Gain)** vs **Gini Impurity**.
- **Hyperparameter Controls**: Configurable Maximum Tree Depth, Minimum Samples Split, and Minimum Samples Leaf.
- **Data Leakage Prevention**: Stratified train/test partitioning with transformations fit strictly on training splits.
- **Evaluation & Diagnostics**: Accuracy, Weighted and Macro-averaged Precision, Recall, and F1-Scores, Cohen's Kappa, ROC-AUC, and Confusion Matrix Heatmap.
- **Interactive Visualization**: Hierarchical Plotly diagram with hover tooltips (showing sample counts, impurity, and class distributions) plus downloadable publication-ready PNG.

### 2. K-Means Clustering (Unsupervised)
- **Feature Selection & Distance Scaling**: Select arbitrary numeric feature subsets with optional standardization (`StandardScaler`) or normalization (`MinMaxScaler`).
- **Interpretable Centroids Table**: Centroids are automatically inverse-transformed and displayed in **original measurement units** (e.g. real dollars and score points).
- **Quality Metrics**: Inertia (SSE) and mathematically guarded **Silhouette Score** ($2 \le K < n$).
- **Elbow Method Analysis**: Evaluates $K \in [2, 10]$ with automated detection of the optimal elbow point.
- **2D/3D PCA Projection**: Principal Component Analysis with explained variance ratio diagnostics.

### 3. Apriori Association Rule Mining
- **Multi-Format Input Support**: Basket lists (comma/semicolon strings), One-Hot binary matrices, Long invoice tables, and discretized categorical records.
- **Mining Engine**: Generates frequent itemsets using the Apriori anti-monotone pruning property.
- **Rule Metrics**: Calculates Support, Confidence, Lift, Leverage, and Conviction.

### 4. FP-Growth (Frequent Pattern Mining) & Side-by-Side Benchmark
- **Candidate-Free Mining**: Compresses transactions into a compact Prefix Tree (FP-Tree) and conditional pattern bases in only 2 database scans.
- **High-Resolution Side-by-Side Benchmark**:
  - Live execution timer using `time.perf_counter()`.
  - Side-by-side table comparing Execution Time (ms), Frequent Itemsets Count, Association Rules Count, and Speedup Ratio.
  - **100% Mathematical Consistency Verification**: Confirms that both Apriori and FP-Growth extract identical itemset solutions under matching support thresholds.

---

## Setup & Run Instructions

### Windows (Local)

1. Open PowerShell or Command Prompt inside the repository folder:
```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

2. Run the Streamlit Application:
```powershell
streamlit run frontend/app.py
```
The application will launch automatically at: `http://localhost:8501`.

3. Run the Automated Test Suite:
```powershell
pytest tests/ -v
```

---

## Deployment to Streamlit Community Cloud

The repository is pre-configured for zero-config deployment on **Streamlit Community Cloud**:
1. Push repository commits to GitHub (`https://github.com/Saiadabala22/DWDM`).
2. Log in to [share.streamlit.io](https://share.streamlit.io/).
3. Click **New app**, select the repository `Saiadabala22/DWDM`, and set:
   - **Main file path**: `frontend/app.py`
4. Click **Deploy**. The application uses SQLite (shipped standard with Python) and auto-initializes `data/dwdm_warehouse.db` on launch.

---

## Provisional Syllabus Coverage Matrix

| Topic | Implemented Feature | Verification Evidence | Status |
| :--- | :--- | :--- | :--- |
| **Data Ingestion** | Multi-format loader (CSV, Excel, ARFF, ZIP) with instant profiling | `profile_dataset()`, Page 1 | Verified |
| **Data Preprocessing** | Missing value imputation, duplicate removal, 1.5x IQR outliers, scaling | `clean()`, `tests/test_preprocessing.py` | Verified |
| **Decision Tree** | Gini & Entropy/Information Gain, depth tuning, stratified split, metrics | `models.py`, `tests/test_classification.py` | Verified |
| **K-Means Clustering** | Feature selection, scaling, readable centroids, SSE, Silhouette, Elbow | `kmeans.py`, `tests/test_clustering.py` | Verified |
| **Apriori Algorithm** | Frequent itemsets and rules from basket/tabular formats | `rules.py`, `tests/test_association.py` | Verified |
| **FP-Growth Algorithm** | FP-Tree frequent pattern mining without candidate generation | `rules.py`, `tests/test_association.py` | Verified |
| **Apriori vs FP-Growth** | High-res benchmark timing, speedup, and mathematical consistency | `compare_algorithms()`, Page 6 | Verified |
| **Data Warehouse & ETL** | SQLite persistence, table row auditing, and schema logging | `etl.py`, `database.py`, Page 3 & 4 | Verified |
| **OLAP Operations** | Roll-up, Drill-down, Slice, Dice, and Pivot tables | `olap.py`, Page 5 | Verified |
| **Visual Analytics** | Interactive Plotly trees, PCA clusters, heatmaps, box/violin plots | `charts.py`, Page 8 | Verified |
| **Evaluation Dashboard**| Unified multi-algorithm diagnostics (Classification/Clustering/ARM) | Page 7 | Verified |
| **Reporting & Export** | Multi-sheet Excel, PDF, HTML, CSV, and JSON downloads | `report.py`, Page 9 | Verified |

---

## Viva Voce Questions & Model Answers

1. **Q: What is the difference between Entropy and Gini Impurity?**
   - **A**: Entropy ($-\sum p_i \log_2 p_i$) measures information disorder and ranges from 0 to $\log_2(C)$. Gini ($1 - \sum p_i^2$) measures the probability of misclassification and ranges from 0 to $1 - 1/C$. Gini avoids logarithmic computation and is faster, while Entropy often builds slightly more balanced trees.

2. **Q: Why does K-Means require feature scaling?**
   - **A**: K-Means uses Euclidean distance. Large-magnitude features (e.g. Annual Income in tens of thousands) dominate small-magnitude features (e.g. Age 18–70). StandardScaler or MinMaxScaler balances feature contributions.

3. **Q: What is the Apriori Property?**
   - **A**: All non-empty subsets of a frequent itemset must also be frequent. Conversely, if an itemset is infrequent, all its supersets are infrequent and can be pruned immediately.

4. **Q: Why is FP-Growth faster than Apriori?**
   - **A**: Apriori scans the database multiple times and generates $C_k$ candidate itemsets. FP-Growth scans the database only twice, builds a compact FP-Tree, and mines frequent patterns recursively using conditional pattern bases without candidate generation.

---

## 5-Minute Professor Demonstration Walkthrough

1. **Ingestion (Page 1)**: Select built-in sample `Groceries Transactions` or `Mall Customers`. Show row counts, missing values, and column types.
2. **Preprocessing (Page 2)**: Demonstrate median numeric imputation, mode categorical imputation, and 1.5x IQR outlier filtering. Inspect before-and-after metrics and download the cleaned CSV.
3. **Data Warehouse (Pages 3 & 4)**: Run ETL to insert records into the SQLite warehouse (`data/dwdm_warehouse.db`). Inspect table schemas on Page 4.
4. **Classification (Page 6, Tab 1)**: Run Decision Tree on `Iris Sample`. Compare Gini vs Entropy, view Accuracy/F1/Confusion Matrix, and expand the **interactive Plotly tree diagram**.
5. **Clustering (Page 6, Tab 3)**: Select `Mall Customers` (`Annual_Income_k`, `Spending_Score`). Run Elbow method to identify optimal $K$, then run K-Means to inspect the **Centroids table in original dollar and score units** and 2D PCA cluster projection.
6. **Association Rules & Benchmark (Page 6, Tab 2)**: Select `Groceries Transactions`. Click **Compare Apriori vs FP-Growth** to show side-by-side timing in milliseconds, verified identical itemsets, and rule metrics (Support, Confidence, Lift).
7. **Evaluation & Report (Pages 7 & 9)**: Open the consolidated Evaluation Dashboard (Page 7) and generate the complete PDF / Multi-Sheet Excel report (Page 9).
