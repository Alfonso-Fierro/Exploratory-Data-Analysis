"""
Parametric hypothesis tests.

Implements t-tests and ANOVA for normally distributed data.
"""

from dataclasses import dataclass
import numpy as np
from scipy import stats
from typing import List
from eda_suite.utils.validators import validate_array


@dataclass
class TTestResult:
    """Result of t-test."""

    statistic: float
    p_value: float
    degrees_freedom: float
    test_name: str
    reject_null: bool


def one_sample_ttest(
    data: np.ndarray,
    population_mean: float
) -> TTestResult:
    """
    One-sample t-test.

    Args:
        data: Sample data
        population_mean: Hypothesized population mean

    Returns:
        TTestResult with test statistics
    """
    arr = validate_array(data)
    statistic, p_value = stats.ttest_1samp(arr, population_mean)

    return TTestResult(
        statistic=float(statistic),
        p_value=float(p_value),
        degrees_freedom=len(arr) - 1,
        test_name="One-Sample T-Test",
        reject_null=p_value < 0.05
    )


def two_sample_ttest(
    sample1: np.ndarray,
    sample2: np.ndarray
) -> TTestResult:
    """
    Independent two-sample t-test.

    Args:
        sample1: First sample
        sample2: Second sample

    Returns:
        TTestResult with test statistics
    """
    arr1 = validate_array(sample1)
    arr2 = validate_array(sample2)

    statistic, p_value = stats.ttest_ind(arr1, arr2)
    df = len(arr1) + len(arr2) - 2

    return TTestResult(
        statistic=float(statistic),
        p_value=float(p_value),
        degrees_freedom=df,
        test_name="Two-Sample T-Test",
        reject_null=p_value < 0.05
    )


def paired_ttest(
    before: np.ndarray,
    after: np.ndarray
) -> TTestResult:
    """
    Paired t-test for related samples.

    Args:
        before: Measurements before treatment
        after: Measurements after treatment

    Returns:
        TTestResult with test statistics
    """
    arr1 = validate_array(before)
    arr2 = validate_array(after)

    statistic, p_value = stats.ttest_rel(arr1, arr2)

    return TTestResult(
        statistic=float(statistic),
        p_value=float(p_value),
        degrees_freedom=len(arr1) - 1,
        test_name="Paired T-Test",
        reject_null=p_value < 0.05
    )
