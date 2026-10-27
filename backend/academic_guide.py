"""Academic Explanations, Syllabus Coverage Matrix, Viva Q&A, and Professor Demonstration Script."""

SYLLABUS_MATRIX = [
    {
        "Topic": "Data Ingestion & Profiling",
        "Implemented Feature": "Multi-format loader (CSV, Excel, ARFF, ZIP) with type inference, shape, missing values, duplicates, and built-in sample datasets",
        "Verification Evidence": "Verified via data_loader.py, profile_dataset(), and tests/test_dataset_loader.py",
        "Status": "Verified"
    },
    {
        "Topic": "Data Preprocessing",
        "Implemented Feature": "Missing-value imputation (median/mean/mode/constant), duplicate removal, IQR outlier detection, one-hot encoding, and StandardScaler / MinMaxScaler",
        "Verification Evidence": "Verified via preprocessing.py, before/after statistics, and tests/test_preprocessing.py",
        "Status": "Verified"
    },
    {
        "Topic": "Decision Tree (Classification)",
        "Implemented Feature": "Decision Tree Classifier with Gini Impurity & Entropy/Information Gain, max depth slider, min samples split/leaf, stratified train-test split, confusion matrix, classification report, feature importances, and interactive Plotly tree diagram",
        "Verification Evidence": "Verified via models.py and tests/test_classification.py",
        "Status": "Verified"
    },
    {
        "Topic": "K-Means (Clustering)",
        "Implemented Feature": "K-Means clustering with numeric feature selection, distance scaling, readable centroids in original units, SSE/Inertia, Silhouette score, 2D PCA projection with explained variance ratio, and Elbow method",
        "Verification Evidence": "Verified via kmeans.py and tests/test_clustering.py",
        "Status": "Verified"
    },
    {
        "Topic": "Apriori (Association Rule Mining)",
        "Implemented Feature": "Apriori algorithm with multi-format transaction parsing (basket list, binary matrix, categorical tokens), candidate generation, pruning, frequent itemsets, and association rules (support, confidence, lift, leverage, conviction)",
        "Verification Evidence": "Verified via rules.py, groceries_transactions.csv, and tests/test_association.py",
        "Status": "Verified"
    },
    {
        "Topic": "FP-Growth (Frequent Pattern Mining)",
        "Implemented Feature": "FP-Growth algorithm using FP-Tree and conditional pattern base, avoiding candidate generation, with identical transaction support and rules extraction",
        "Verification Evidence": "Verified via rules.py and tests/test_association.py",
        "Status": "Verified"
    },
    {
        "Topic": "Apriori vs FP-Growth Benchmark",
        "Implemented Feature": "High-resolution side-by-side timer (perf_counter), frequent itemset count, rule count, speedup ratio, consistency verification, and architectural trade-off analysis",
        "Verification Evidence": "Verified via compare_algorithms() and tests/test_apriori_fpgrowth_comparison.py",
        "Status": "Verified"
    },
    {
        "Topic": "ETL & SQLite Data Warehouse",
        "Implemented Feature": "Extract-Transform-Load pipeline persisting cleaned records and column schemas into SQLite warehouse (dwdm_warehouse.db) with transaction logging",
        "Verification Evidence": "Verified via etl.py, database.py, and SQLite table inspection",
        "Status": "Verified"
    },
    {
        "Topic": "OLAP Operations",
        "Implemented Feature": "Interactive Roll-up, Drill-down, Slice, Dice, and Pivot tables on warehouse data",
        "Verification Evidence": "Verified via backend/olap/olap.py and frontend/pages/5_OLAP.py",
        "Status": "Verified"
    },
    {
        "Topic": "Data Visualization & Exploration",
        "Implemented Feature": "Distributions (Histogram, Box, Violin, Density), Categories (Bar, Pie, Stacked), Relationships (Correlation Heatmap, Scatter, Line, 3D Scatter, Parallel Coordinates), and Mining Visuals",
        "Verification Evidence": "Verified via backend/visualization/charts.py and frontend/pages/8_Visualization.py",
        "Status": "Verified"
    },
    {
        "Topic": "Evaluation Dashboard",
        "Implemented Feature": "Unified dashboard reporting Classification metrics (Accuracy, Precision, Recall, F1, Kappa, AUC), Clustering diagnostics (Inertia, Silhouette, Centroids), and Association metrics",
        "Verification Evidence": "Verified via frontend/pages/7_Evaluation.py",
        "Status": "Verified"
    },
    {
        "Topic": "Final Report & Export",
        "Implemented Feature": "Multi-format report generation (PDF, HTML, Excel, CSV, JSON) encapsulating dataset profiling and mining metrics",
        "Verification Evidence": "Verified via report.py and frontend/pages/9_Final_Report.py",
        "Status": "Verified"
    }
]

VIVA_QA = [
    {
        "q": "What is the difference between Entropy and Gini Impurity in Decision Trees?",
        "a": "Entropy is based on information theory: Entropy = -sum(p_i * log2(p_i)). It measures the uncertainty/disorder in a node and ranges from 0 (pure) to log2(C). Gini Impurity measures the probability of misclassifying a randomly chosen element: Gini = 1 - sum(p_i^2), ranging from 0 to 1 - 1/C. Gini is computationally faster because it avoids logarithmic calculations, while Entropy yields slightly more balanced trees."
    },
    {
        "q": "What is Information Gain?",
        "a": "Information Gain = Entropy(Parent) - sum((|D_v| / |D|) * Entropy(Child_v)). It measures the reduction in entropy achieved by partitioning the dataset on a given attribute. Decision Tree algorithms like ID3 select the attribute that maximizes Information Gain."
    },
    {
        "q": "How does J48 differ from ID3?",
        "a": "ID3 only supports categorical attributes, has no tree pruning, and is prone to overfitting. J48 (Java implementation of C4.5) supports both continuous and discrete attributes, handles missing values, and implements post-pruning (using pessimistic pruning or cost-complexity pruning) to prevent overfitting."
    },
    {
        "q": "Why is feature scaling essential before K-Means clustering?",
        "a": "K-Means uses Euclidean distance (d = sqrt(sum((x_i - y_i)^2))). If one feature has values from 10,000 to 100,000 (like Annual Income) and another ranges from 1 to 5 (like Ratings), the high-magnitude feature will completely dominate the distance calculations. Standardization (z-score) or Min-Max scaling ensures every feature contributes equally."
    },
    {
        "q": "What is the Silhouette Score, and how is it interpreted?",
        "a": "For a sample i: s(i) = (b(i) - a(i)) / max(a(i), b(i)), where a(i) is the mean intra-cluster distance and b(i) is the mean nearest-cluster distance. Scores range from -1 to +1. +1 indicates well-clustered samples far from neighbors, 0 indicates overlapping cluster boundaries, and negative values indicate incorrect cluster assignment."
    },
    {
        "q": "What is the Apriori Property?",
        "a": "The Apriori Property states: 'All non-empty subsets of a frequent itemset must also be frequent.' Conversely (the anti-monotone property), if an itemset is infrequent, all of its supersets are guaranteed to be infrequent and can be pruned immediately without scanning the database."
    },
    {
        "q": "Why is FP-Growth generally faster than Apriori?",
        "a": "Apriori requires multiple full database scans (one scan per itemset length k) and generates huge numbers of candidate itemsets (C_k). In contrast, FP-Growth scans the database only twice, compresses the transactions into a compact Prefix Tree (FP-Tree), and extracts frequent itemsets directly via recursive conditional pattern bases without candidate generation."
    },
    {
        "q": "What is the difference between Confidence and Lift in association rules?",
        "a": "Confidence(A -> B) = Support(A U B) / Support(A). It measures how often B appears when A is present. Lift(A -> B) = Confidence(A -> B) / Support(B). It measures how much more often A and B occur together than would be expected if they were statistically independent. Lift = 1 means independent; Lift > 1 indicates positive association; Lift < 1 indicates negative association."
    },
    {
        "q": "How do you avoid data leakage during preprocessing in machine learning?",
        "a": "Transformations like mean imputation, standardization, and label encoding must be computed ('fit') ONLY on the training dataset (X_train), and then applied ('transform') to both the training and test sets. Fitting on the entire dataset leaks information about test set distribution, leading to overly optimistic evaluation metrics."
    },
    {
        "q": "What is the difference between OLTP and OLAP?",
        "a": "OLTP (Online Transaction Processing) is optimized for day-to-day transactional inserts and updates with normalized schemas (3NF) and low latency. OLAP (Online Analytical Processing) is optimized for complex aggregation, reporting, and ad-hoc queries over historical data using denormalized multidimensional schemas (Star or Snowflake)."
    }
]

DEMO_SCRIPT = """
### Professor Demonstration Walkthrough (5-Minute Script)

1. **Step 1: Dataset Ingestion (Page 1)**
   - Click the **Built-in Sample Datasets** dropdown and select `Groceries Transactions (Market Basket)`.
   - Show the row count (50 transactions), column types, and instant summary metrics.
   - Demonstrate the same for `Mall Customers (Clustering)` and `Iris Sample (Classification)`.

2. **Step 2: Data Preprocessing (Page 2)**
   - Select `Titanic / Mall Customers` and configure:
     - Impute missing values (median for numeric, mode for categorical).
     - Remove duplicates.
     - Outlier detection (1.5x IQR rule).
   - Click **Run Preprocessing** to inspect the live before/after diagnostic summary.

3. **Step 3: ETL & SQLite Data Warehouse (Pages 3 & 4)**
   - Click **Extract → Transform → Load** to persist the cleaned data into `data/dwdm_warehouse.db`.
   - Navigate to Page 4 (Warehouse) to show the schema metadata and SQLite table rows.

4. **Step 4: Classification - Decision Tree (Page 6, Tab 1)**
   - Load `Iris Sample` or `Processed Dataset`, select Target = `species`, Algorithm = `Decision Tree`.
   - Set Criterion to `Entropy (Information Gain)` or `Gini Impurity`, adjust Max Depth.
   - Run Classification: point out Accuracy, Macro F1, Confusion Matrix, and the **interactive Plotly Tree Diagram** (hover over nodes to view sample distributions and split thresholds).

5. **Step 5: Clustering - K-Means (Page 6, Tab 3)**
   - Load `Mall Customers`, select `Annual_Income_k` and `Spending_Score`.
   - Run the **Elbow Method**: show the SSE curve and the detected optimal K.
   - Run K-Means with K=5: show the **readable Centroid Table in original dollar and score units**, Silhouette score, and 2D PCA cluster visualization.

6. **Step 6: Association Rule Mining & Apriori vs FP-Growth Benchmark (Page 6, Tab 2)**
   - Select `Groceries Transactions`.
   - Set Minimum Support = 0.10, Minimum Confidence = 0.40.
   - Click **Compare Apriori vs FP-Growth**:
     - Point out the side-by-side high-resolution execution time (ms).
     - Verify that both algorithms produce identical frequent itemsets.
     - Show the resulting association rules with Support, Confidence, and Lift.

7. **Step 7: Evaluation & Final Reports (Pages 7 & 9)**
   - Open Page 7 (Evaluation Dashboard) for the consolidated metrics across all stages.
   - Open Page 9 (Final Report) and download the comprehensive report in **PDF, HTML, and Excel**.
"""
