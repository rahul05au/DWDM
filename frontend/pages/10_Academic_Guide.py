import streamlit as st
import pandas as pd
from utils.ui import load_css
from backend.academic_guide import SYLLABUS_MATRIX, VIVA_QA, DEMO_SCRIPT

load_css()
st.title('10. Academic Concepts & Syllabus Matrix')
st.caption('Comprehensive academic reference, provisional syllabus coverage matrix, viva preparation, and professor demonstration guide.')

t1, t2, t3, t4 = st.tabs(['Syllabus Coverage Matrix', 'Core Algorithm Concepts', 'Viva Voce Questions & Answers', 'Professor Demonstration Script'])

# =========================================================================
# TAB 1: SYLLABUS COVERAGE MATRIX
# =========================================================================
with t1:
    st.subheader('Provisional Syllabus Coverage Matrix')
    st.markdown("This matrix tracks the university Data Mining curriculum against verified, operational code implementations:")

    matrix_df = pd.DataFrame(SYLLABUS_MATRIX)
    st.dataframe(matrix_df, use_container_width=True)

# =========================================================================
# TAB 2: CORE CONCEPTS
# =========================================================================
with t2:
    st.subheader('Theoretical Data Mining Foundations')

    with st.expander('1. Decision Tree Classification (ID3, J48, CART)', expanded=True):
        st.markdown(r"""
        - **Classification Goal**: Map input attributes $\mathbf{x} \in \mathbb{R}^d$ to a discrete target class $y \in \{C_1, \dots, C_k\}$.
        - **Entropy**: Quantifies node disorder:
          $$\text{Entropy}(D) = -\sum_{i=1}^{k} p_i \log_2(p_i)$$
        - **Information Gain**: Reduction in entropy from splitting on feature $A$:
          $$\text{Gain}(D, A) = \text{Entropy}(D) - \sum_{v \in \text{Values}(A)} \frac{|D_v|}{|D|} \text{Entropy}(D_v)$$
        - **Gini Impurity (CART)**:
          $$\text{Gini}(D) = 1 - \sum_{i=1}^{k} p_i^2$$
        - **Overfitting & Pruning**: Unconstrained trees memorize training noise ($100\%$ train accuracy, low test accuracy). Pre-pruning limits `max_depth` and `min_samples_leaf`; post-pruning removes subtrees using cost-complexity pruning ($\alpha$).
        """)

    with st.expander('2. K-Means Clustering'):
        st.markdown(r"""
        - **Objective**: Partition $n$ unlabeled points into $K$ clusters $S = \{S_1, \dots, S_K\}$ minimizing Within-Cluster Sum of Squares (Inertia/SSE):
          $$\text{SSE} = \sum_{k=1}^{K} \sum_{\mathbf{x} \in S_k} \|\mathbf{x} - \boldsymbol{\mu}_k\|^2$$
        - **Standard Lloyd's Algorithm**:
          1. Initialize $K$ centroids $\boldsymbol{\mu}_1, \dots, \boldsymbol{\mu}_K$ (e.g. k-means++).
          2. **Assignment Step**: Assign each $\mathbf{x}_i$ to the nearest centroid using Euclidean distance.
          3. **Update Step**: Recompute centroids: $\boldsymbol{\mu}_k = \frac{1}{|S_k|} \sum_{\mathbf{x} \in S_k} \mathbf{x}$.
          4. Repeat until convergence ($\Delta \boldsymbol{\mu} < \epsilon$).
        - **Elbow Method**: Plots SSE vs $K$. The 'elbow' point marks diminishing returns where adding another cluster yields marginal variance explanation.
        - **Silhouette Analysis**: Evaluates separation distance between clusters, ranging from $-1$ to $+1$.
        """)

    with st.expander('3. Apriori Association Rule Mining'):
        st.markdown(r"""
        - **Market Basket Model**: Discovers relationships of the form $X \Rightarrow Y$ where $X \cap Y = \emptyset$.
        - **Support**:
          $$\text{Support}(X \Rightarrow Y) = P(X \cup Y) = \frac{\text{Count}(X \cup Y)}{N}$$
        - **Confidence**:
          $$\text{Confidence}(X \Rightarrow Y) = P(Y | X) = \frac{\text{Support}(X \cup Y)}{\text{Support}(X)}$$
        - **Lift**:
          $$\text{Lift}(X \Rightarrow Y) = \frac{P(X \cup Y)}{P(X) \cdot P(Y)} = \frac{\text{Confidence}(X \Rightarrow Y)}{\text{Support}(Y)}$$
          - $\text{Lift} = 1$: Independent.
          - $\text{Lift} > 1$: Positive correlation (buying $X$ boosts probability of buying $Y$).
        - **The Apriori Property**: Any subset of a frequent itemset must also be frequent. Infrequent subsets are pruned early to constrain search space.
        """)

    with st.expander('4. FP-Growth (Frequent Pattern Growth)'):
        st.markdown(r"""
        - **Philosophy**: Avoids candidate generation entirely ($C_k$).
        - **Methodology**:
          1. **Database Scan 1**: Count support of 1-itemsets; filter infrequent items; sort frequent items in descending frequency order.
          2. **Database Scan 2**: Insert each transaction into an **FP-Tree** (Frequent Pattern Tree) with shared prefix paths, maintaining a Header Table with linked lists for each item.
          3. **Divide and Conquer**: For each item in the header table (bottom-up), build its *Conditional Pattern Base*, construct a *Conditional FP-Tree*, and recursively extract frequent patterns.
        - **Advantage over Apriori**: Dramatic speedup on dense datasets and lower support thresholds because no candidate generation or repetitive disk scans are needed.
        """)

# =========================================================================
# TAB 3: VIVA VOCE QUESTIONS & ANSWERS
# =========================================================================
with t3:
    st.subheader('Top Viva Voce Questions & Answers')
    st.caption('Model answers for internal and external lab examiners:')

    for i, item in enumerate(VIVA_QA, 1):
        with st.expander(f"Q{i}: {item['q']}", expanded=(i <= 3)):
            st.markdown(f"**Answer:** {item['a']}")

# =========================================================================
# TAB 4: DEMONSTRATION SCRIPT
# =========================================================================
with t4:
    st.subheader('Professor Demonstration Guide')
    st.markdown(DEMO_SCRIPT)
