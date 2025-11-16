"""
Multivariate analysis module for EDA Suite.

Provides tools for analyzing multiple variables simultaneously.
"""

from eda_suite.multivariate.clustering import (
    kmeans_analysis,
    hierarchical_clustering
)
from eda_suite.multivariate.outliers import (
    detect_multivariate_outliers,
    mahalanobis_distance
)

__all__ = [
    "kmeans_analysis",
    "hierarchical_clustering",
    "detect_multivariate_outliers",
    "mahalanobis_distance"
]
