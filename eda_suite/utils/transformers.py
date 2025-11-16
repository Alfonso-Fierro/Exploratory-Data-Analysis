"""
Data transformation utilities.

Provides standardization and normalization functions.
"""

import numpy as np
from eda_suite.utils.validators import validate_array


def standardize(data: np.ndarray) -> np.ndarray:
    """
    Standardize data to mean=0 and std=1.

    Args:
        data: Input array

    Returns:
        Standardized array
    """
    arr = validate_array(data)
    mean = np.mean(arr)
    std = np.std(arr, ddof=1)

    if std == 0:
        return np.zeros_like(arr)

    return (arr - mean) / std


def normalize(data: np.ndarray) -> np.ndarray:
    """
    Normalize data to range [0, 1].

    Args:
        data: Input array

    Returns:
        Normalized array
    """
    arr = validate_array(data)
    min_val = np.min(arr)
    max_val = np.max(arr)

    if max_val == min_val:
        return np.zeros_like(arr)

    return (arr - min_val) / (max_val - min_val)
