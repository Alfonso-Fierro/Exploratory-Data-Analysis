"""
Non-parametric hypothesis tests.

Implements distribution-free statistical tests.
"""

from dataclasses import dataclass
import numpy as np
from scipy import stats
from typing import List
from eda_suite.utils.validators import validate_array


@dataclass
class NonparametricResult:
    """Result of non-parametric test."""

    statistic: float
    p_value: float
    test_name: str
    reject_null: bool


def mann_whitney_test(
    sample1: np.ndarray,
    sample2: np.ndarray
) -> NonparametricResult:
    """
    Mann-Whitney U test (Wilcoxon rank-sum test).

    Args:
        sample1: First independent sample
        sample2: Second independent sample

    Returns:
        NonparametricResult with test statistics
    """
    arr1 = validate_array(sample1)
    arr2 = validate_array(sample2)

    statistic, p_value = stats.mannwhitneyu(arr1, arr2)

    return NonparametricResult(
        statistic=float(statistic),
        p_value=float(p_value),
        test_name="Mann-Whitney U",
        reject_null=p_value < 0.05
    )


def wilcoxon_test(
    before: np.ndarray,
    after: np.ndarray
) -> NonparametricResult:
    """
    Wilcoxon signed-rank test for paired samples.

    Args:
        before: Measurements before treatment
        after: Measurements after treatment

    Returns:
        NonparametricResult with test statistics
    """
    arr1 = validate_array(before)
    arr2 = validate_array(after)

    statistic, p_value = stats.wilcoxon(arr1, arr2)

    return NonparametricResult(
        statistic=float(statistic),
        p_value=float(p_value),
        test_name="Wilcoxon Signed-Rank",
        reject_null=p_value < 0.05
    )


def kruskal_wallis_test(groups: List[np.ndarray]) -> NonparametricResult:
    """
    Kruskal-Wallis H-test for independent samples.

    Args:
        groups: List of independent sample groups

    Returns:
        NonparametricResult with test statistics
    """
    statistic, p_value = stats.kruskal(*groups)

    return NonparametricResult(
        statistic=float(statistic),
        p_value=float(p_value),
        test_name="Kruskal-Wallis H",
        reject_null=p_value < 0.05
    )


def friedman_test(groups: List[np.ndarray]) -> NonparametricResult:
    """
    Friedman test for repeated measures.

    Args:
        groups: List of related sample groups

    Returns:
        NonparametricResult with test statistics
    """
    statistic, p_value = stats.friedmanchisquare(*groups)

    return NonparametricResult(
        statistic=float(statistic),
        p_value=float(p_value),
        test_name="Friedman",
        reject_null=p_value < 0.05
    )
