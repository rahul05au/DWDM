import pytest
import pandas as pd
import numpy as np
from backend.clustering.kmeans import run, elbow


@pytest.fixture
def sample_clustering_data():
    np.random.seed(42)
    # Create two clear clusters
    c1 = np.random.normal(loc=[10, 10], scale=1, size=(25, 2))
    c2 = np.random.normal(loc=[50, 50], scale=1, size=(25, 2))
    data = np.vstack([c1, c2])
    return pd.DataFrame(data, columns=['Income', 'Score'])


def test_kmeans_execution(sample_clustering_data):
    res = run(sample_clustering_data, k=2, scale='standard')
    assert res['k'] == 2
    assert len(res['labels']) == len(sample_clustering_data)
    assert res['inertia'] > 0
    # Silhouette should be strong (> 0.5) for separated clusters
    assert res['silhouette'] is not None and res['silhouette'] > 0.5
    # Centroids table must have 2 rows and matching feature columns
    assert res['centroids'].shape == (2, 2)
    assert list(res['centroids'].columns) == ['Income', 'Score']


def test_kmeans_centroids_original_units(sample_clustering_data):
    res = run(sample_clustering_data, k=2, scale='standard')
    centroids = res['centroids']
    # Centroid values in original units should be around ~10 and ~50
    means = sorted(centroids['Income'].values)
    assert 8.0 < means[0] < 12.0
    assert 48.0 < means[1] < 52.0


def test_elbow_method(sample_clustering_data):
    e_df, rec_k = elbow(sample_clustering_data, kmax=5)
    assert len(e_df) >= 3
    assert 'K' in e_df.columns
    assert 'SSE' in e_df.columns
    assert 'Silhouette' in e_df.columns
    # SSE must decrease monotonically as K increases
    sse_vals = e_df['SSE'].values
    for i in range(len(sse_vals) - 1):
        assert sse_vals[i] >= sse_vals[i + 1]
    assert rec_k in [2, 3, 4, 5]
