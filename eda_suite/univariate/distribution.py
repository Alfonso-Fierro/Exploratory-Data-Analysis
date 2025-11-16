"""
Distribution fitting and testing.

Fits theoretical distributions to empirical data.
"""

import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import Tuple
from eda_suite.utils.validators import validate_array


@dataclass
class DistributionFit:
    """Result of distribution fitting."""

    distribution_name: str
    parameters: Tuple
    ks_statistic: float
    p_value: float
    is_good_fit: bool


def fit_distribution(
    data: np.ndarray,
    distribution: str = "norm"
) -> DistributionFit:
    """
    Fit theoretical distribution to data.

    Args:
        data: Input array
        distribution: Distribution name (norm, expon, gamma, etc.)

    Returns:
        DistributionFit with parameters and fit statistics
    """
    arr = validate_array(data)

    dist = getattr(stats, distribution)
    params = dist.fit(arr)

    ks_stat, p_val = stats.kstest(arr, distribution, args=params)

    return DistributionFit(
        distribution_name=distribution,
        parameters=params,
        ks_statistic=float(ks_stat),
        p_value=float(p_val),
        is_good_fit=p_val > 0.05
    )


def test_distribution_fit(data: np.ndarray) -> dict:
    """
    Test multiple distributions.

    Args:
        data: Input array

    Returns:
        Dictionary with fit results for multiple distributions
    """
    distributions = ["norm", "expon", "gamma", "lognorm", "weibull_min"]
    results = {}

    for dist in distributions:
        try:
            results[dist] = fit_distribution(data, dist)
        except:
            continue

    return results
