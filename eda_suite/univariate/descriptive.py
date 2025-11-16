"""
Descriptive statistics for univariate data.

Computes measures of central tendency and dispersion.
"""

import numpy as np
from scipy import stats
from dataclasses import dataclass
from eda_suite.utils.validators import validate_array


@dataclass
class DescriptiveStats:
    """Descriptive statistics summary."""

    mean: float
    median: float
    mode: float
    std: float
    variance: float
    min: float
    max: float
    range: float
    iqr: float


def compute_statistics(data: np.ndarray) -> DescriptiveStats:
    """
    Compute comprehensive descriptive statistics.

    Args:
        data: Input array

    Returns:
        DescriptiveStats with all measures
    """
    arr = validate_array(data)

    q1, q3 = np.percentile(arr, [25, 75])
    mode_result = stats.mode(arr, keepdims=True)

    return DescriptiveStats(
        mean=float(np.mean(arr)),
        median=float(np.median(arr)),
        mode=float(mode_result.mode[0]),
        std=float(np.std(arr, ddof=1)),
        variance=float(np.var(arr, ddof=1)),
        min=float(np.min(arr)),
        max=float(np.max(arr)),
        range=float(np.ptp(arr)),
        iqr=float(q3 - q1)
    )


@dataclass
class MomentStats:
    """Higher moment statistics."""

    skewness: float
    kurtosis: float
    excess_kurtosis: float


def compute_moments(data: np.ndarray) -> MomentStats:
    """
    Compute skewness and kurtosis.

    Args:
        data: Input array

    Returns:
        MomentStats with moment measures
    """
    arr = validate_array(data)

    skew = stats.skew(arr)
    kurt = stats.kurtosis(arr, fisher=False)

    return MomentStats(
        skewness=float(skew),
        kurtosis=float(kurt),
        excess_kurtosis=float(kurt - 3)
    )


def compute_quantiles(data: np.ndarray) -> dict:
    """
    Compute various quantiles.

    Args:
        data: Input array

    Returns:
        Dictionary with quantile values
    """
    arr = validate_array(data)

    quantiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    values = np.percentile(arr, [q * 100 for q in quantiles])

    return {f"q{int(q*100):02d}": float(v) for q, v in zip(quantiles, values)}
