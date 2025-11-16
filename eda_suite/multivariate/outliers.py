"""
Multivariate outlier detection.

Identifies outliers using multivariate methods.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import mahalanobis as scipy_mahalanobis


def mahalanobis_distance(
    data: np.ndarray,
    robust: bool = False
) -> np.ndarray:
    """
    Compute Mahalanobis distance for each observation.

    Args:
        data: Feature matrix
        robust: Use robust estimates

    Returns:
        Array of Mahalanobis distances
    """
    mean = np.mean(data, axis=0)
    cov = np.cov(data, rowvar=False)

    inv_cov = np.linalg.inv(cov)
    distances = []

    for obs in data:
        dist = scipy_mahalanobis(obs, mean, inv_cov)
        distances.append(dist)

    return np.array(distances)


def detect_multivariate_outliers(
    data: np.ndarray,
    threshold: float = 3.0
) -> dict:
    """
    Detect multivariate outliers using Mahalanobis distance.

    Args:
        data: Feature matrix
        threshold: Distance threshold for outliers

    Returns:
        Dictionary with outlier information
    """
    distances = mahalanobis_distance(data)
    outliers = distances > threshold

    return {
        "distances": distances,
        "outlier_indices": np.where(outliers)[0],
        "n_outliers": int(np.sum(outliers)),
        "outlier_percentage": float(np.mean(outliers) * 100)
    }
