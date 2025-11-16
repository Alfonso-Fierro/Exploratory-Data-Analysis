"""
Clustering analysis for multivariate data.

Implements k-means and hierarchical clustering.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from dataclasses import dataclass


@dataclass
class ClusteringResult:
    """Result of clustering analysis."""

    labels: np.ndarray
    n_clusters: int
    inertia: float
    centroids: np.ndarray


def kmeans_analysis(
    data: np.ndarray,
    n_clusters: int = 3
) -> ClusteringResult:
    """
    Perform k-means clustering.

    Args:
        data: Feature matrix
        n_clusters: Number of clusters

    Returns:
        ClusteringResult with cluster assignments
    """
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(data)

    return ClusteringResult(
        labels=labels,
        n_clusters=n_clusters,
        inertia=float(model.inertia_),
        centroids=model.cluster_centers_
    )


def hierarchical_clustering(
    data: np.ndarray,
    n_clusters: int = 3
) -> np.ndarray:
    """
    Perform hierarchical clustering.

    Args:
        data: Feature matrix
        n_clusters: Number of clusters

    Returns:
        Cluster labels
    """
    model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(data)

    return labels
