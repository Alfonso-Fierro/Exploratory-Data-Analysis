"""
Normality tests for distribution analysis.

Implements multiple normality tests following clean code principles.
"""

from dataclasses import dataclass
import numpy as np
from scipy import stats
from eda_suite.utils.validators import validate_array


@dataclass
class TestResult:
    """Result of a statistical test."""

    statistic: float
    p_value: float
    test_name: str
    is_significant: bool


def shapiro_test(data: np.ndarray) -> TestResult:
    """
    Shapiro-Wilk test for normality.

    Args:
        data: Array of sample data

    Returns:
        TestResult with test statistics
    """
    arr = validate_array(data)
    statistic, p_value = stats.shapiro(arr)

    return TestResult(
        statistic=float(statistic),
        p_value=float(p_value),
        test_name="Shapiro-Wilk",
        is_significant=p_value < 0.05
    )


def anderson_test(data: np.ndarray) -> dict:
    """
    Anderson-Darling test for normality.

    Args:
        data: Array of sample data

    Returns:
        Dictionary with test results
    """
    arr = validate_array(data)
    result = stats.anderson(arr)

    return {
        "statistic": float(result.statistic),
        "critical_values": result.critical_values.tolist(),
        "significance_levels": result.significance_level.tolist(),
        "test_name": "Anderson-Darling"
    }


def jarque_bera_test(data: np.ndarray) -> TestResult:
    """
    Jarque-Bera test for normality.

    Args:
        data: Array of sample data

    Returns:
        TestResult with test statistics
    """
    arr = validate_array(data)
    statistic, p_value = stats.jarque_bera(arr)

    return TestResult(
        statistic=float(statistic),
        p_value=float(p_value),
        test_name="Jarque-Bera",
        is_significant=p_value < 0.05
    )


def kolmogorov_smirnov_test(data: np.ndarray) -> TestResult:
    """
    Kolmogorov-Smirnov test for normality.

    Args:
        data: Array of sample data

    Returns:
        TestResult with test statistics
    """
    arr = validate_array(data)
    statistic, p_value = stats.kstest(arr, 'norm')

    return TestResult(
        statistic=float(statistic),
        p_value=float(p_value),
        test_name="Kolmogorov-Smirnov",
        is_significant=p_value < 0.05
    )
