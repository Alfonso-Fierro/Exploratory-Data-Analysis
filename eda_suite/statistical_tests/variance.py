"""
Variance homogeneity tests.

Tests for equality of variances across groups.
"""

from dataclasses import dataclass
import numpy as np
from scipy import stats
from typing import List


@dataclass
class VarianceTestResult:
    """Result of variance homogeneity test."""

    statistic: float
    p_value: float
    test_name: str
    variances_equal: bool


def levene_test(groups: List[np.ndarray]) -> VarianceTestResult:
    """
    Levene test for equality of variances.

    Args:
        groups: List of sample groups

    Returns:
        VarianceTestResult with test statistics
    """
    statistic, p_value = stats.levene(*groups)

    return VarianceTestResult(
        statistic=float(statistic),
        p_value=float(p_value),
        test_name="Levene",
        variances_equal=p_value > 0.05
    )


def bartlett_test(groups: List[np.ndarray]) -> VarianceTestResult:
    """
    Bartlett test for equality of variances.

    Args:
        groups: List of sample groups

    Returns:
        VarianceTestResult with test statistics
    """
    statistic, p_value = stats.bartlett(*groups)

    return VarianceTestResult(
        statistic=float(statistic),
        p_value=float(p_value),
        test_name="Bartlett",
        variances_equal=p_value > 0.05
    )


def fligner_test(groups: List[np.ndarray]) -> VarianceTestResult:
    """
    Fligner-Killeen test for equality of variances.

    Args:
        groups: List of sample groups

    Returns:
        VarianceTestResult with test statistics
    """
    statistic, p_value = stats.fligner(*groups)

    return VarianceTestResult(
        statistic=float(statistic),
        p_value=float(p_value),
        test_name="Fligner-Killeen",
        variances_equal=p_value > 0.05
    )
