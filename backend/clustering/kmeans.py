"""K-Means Clustering module with Centroids in original units, Elbow method, and Silhouette analysis."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def prepare_clustering_data(df: pd.DataFrame, feature_cols=None, scale: str = 'standard'):
    """Prepare numeric matrix for clustering with imputation and optional scaling.

    Args:
        df: Input DataFrame
        feature_cols: List of column names to use. If None, selects all numeric columns.
        scale: 'standard', 'minmax', or 'none'

    Returns:
        X_scaled: np.ndarray used for fitting KMeans
        X_imputed: np.ndarray in original feature units
        feature_names: list of feature names used
        scaler: scaler object used (or None)
    """
    if df is None or df.empty:
        raise ValueError("Dataset is empty.")

    if feature_cols:
        sub_df = df[feature_cols].copy()
    else:
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        if not num_cols:
            # Try to one-hot encode categorical if no numeric columns
            sub_df = pd.get_dummies(df, dtype=float)
        else:
            sub_df = df[num_cols].copy()

    if sub_df.shape[1] == 0:
        raise ValueError("No valid features found for clustering.")

    feature_names = list(sub_df.columns)
    # Median imputation for missing values
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(sub_df)

    if scale == 'standard':
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)
    elif scale == 'minmax':
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_imputed)
    else:
        scaler = None
        X_scaled = X_imputed.copy()

    return X_scaled, X_imputed, feature_names, scaler


def calculate_silhouette(X, labels):
    """Safely calculate Silhouette score only when mathematically valid."""
    unique_labels = set(labels)
    if len(unique_labels) < 2 or len(X) <= len(unique_labels):
        return None
    try:
        sample_sz = min(len(X), 5000)
        return float(silhouette_score(X, labels, sample_size=sample_sz, random_state=42))
    except Exception:
        return None


def run(df: pd.DataFrame, k: int = 3, feature_cols=None, scale: str = 'standard'):
    """Fit K-Means clustering and return model, projections, labels, centroids, and metrics."""
    X_scaled, X_raw, feature_names, scaler = prepare_clustering_data(df, feature_cols=feature_cols, scale=scale)

    n_samples = len(X_scaled)
    if n_samples < 3:
        raise ValueError("Need at least 3 samples to perform clustering.")

    k = max(2, min(int(k), n_samples - 1))

    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = model.fit_predict(X_scaled)

    # 1. Cluster Centroids
    # Centroids in original feature units
    if scaler is not None and hasattr(scaler, 'inverse_transform'):
        raw_centroids = scaler.inverse_transform(model.cluster_centers_)
    else:
        raw_centroids = model.cluster_centers_

    centroids_df = pd.DataFrame(
        raw_centroids,
        columns=feature_names,
        index=[f"Cluster {i}" for i in range(k)]
    )

    # Centroids in scaled units (for comparison)
    scaled_centroids_df = pd.DataFrame(
        model.cluster_centers_,
        columns=feature_names,
        index=[f"Cluster {i}" for i in range(k)]
    )

    # 2. PCA Projection (2D and 3D)
    n_components = min(3, X_scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    p = pca.fit_transform(X_scaled)
    if p.shape[1] == 1:
        p = np.column_stack([p[:, 0], np.zeros(len(p))])

    explained_var = pca.explained_variance_ratio_

    # 3. Cluster Profile (Distribution)
    counts = pd.Series(labels).value_counts().sort_index()
    profile_df = pd.DataFrame({
        'Cluster': [f"Cluster {i}" for i in counts.index],
        'Count': counts.values,
        'Percentage': (counts.values / n_samples * 100).round(2)
    })

    # 4. Metrics
    sil_score = calculate_silhouette(X_scaled, labels)
    inertia = float(model.inertia_)

    return {
        'model': model,
        'labels': labels,
        'p': p,
        'pca': pca,
        'explained_variance': explained_var,
        'silhouette': sil_score,
        'inertia': inertia,
        'centroids': centroids_df,
        'scaled_centroids': scaled_centroids_df,
        'profile': profile_df,
        'feature_names': feature_names,
        'k': k,
        'scale_used': scale
    }


def elbow(df: pd.DataFrame, kmax: int = 10, feature_cols=None, scale: str = 'standard'):
    """Compute Inertia (SSE) and Silhouette scores for K in [2, kmax]."""
    X_scaled, _, _, _ = prepare_clustering_data(df, feature_cols=feature_cols, scale=scale)
    n_samples = len(X_scaled)
    max_k = min(int(kmax), n_samples - 1)

    rows = []
    for k in range(2, max_k + 1):
        m = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_scaled)
        sil = calculate_silhouette(X_scaled, m.labels_)
        rows.append({
            'K': k,
            'SSE': float(m.inertia_),
            'Silhouette': sil
        })

    elbow_df = pd.DataFrame(rows)

    # Simple heuristic to identify elbow (maximum distance from line connecting first & last points)
    if len(elbow_df) >= 3:
        p1 = np.array([elbow_df['K'].iloc[0], elbow_df['SSE'].iloc[0]])
        p2 = np.array([elbow_df['K'].iloc[-1], elbow_df['SSE'].iloc[-1]])
        # 2D cross product magnitude: |(x2-x1)(y1-y0) - (y2-y1)(x1-x0)|
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        norm = np.hypot(dx, dy)
        distances = []
        for _, row in elbow_df.iterrows():
            p0 = np.array([row['K'], row['SSE']])
            d = np.abs(dx * (p1[1] - p0[1]) - dy * (p1[0] - p0[0])) / (norm if norm > 0 else 1.0)
            distances.append(d)
        best_k_idx = int(np.argmax(distances))
        recommended_k = int(elbow_df['K'].iloc[best_k_idx])
    else:
        recommended_k = 3

    return elbow_df, recommended_k
