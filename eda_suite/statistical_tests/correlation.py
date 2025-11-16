"""
Correlation tests for relationship analysis.

Implements parametric and non-parametric correlation tests.
"""

from dataclasses import dataclass
import numpy as np
from scipy import stats
from eda_suite.utils.validators import validate_array


@dataclass
class CorrelationResult:
    """Result of correlation test."""

    coefficient: float
    p_value: float
    test_name: str
    is_significant: bool


def pearson_test(x: np.ndarray, y: np.ndarray) -> CorrelationResult:
    """
    Pearson correlation coefficient test.

    Args:
        x: First variable
        y: Second variable

    Returns:
        CorrelationResult with test statistics
    """
    x_arr = validate_array(x)
    y_arr = validate_array(y)

    coefficient, p_value = stats.pearsonr(x_arr, y_arr)

    return CorrelationResult(
        coefficient=float(coefficient),
        p_value=float(p_value),
        test_name="Pearson",
        is_significant=p_value < 0.05
    )


def spearman_test(x: np.ndarray, y: np.ndarray) -> CorrelationResult:
    """
    Spearman rank correlation coefficient test.

    Args:
        x: First variable
        y: Second variable

    Returns:
        CorrelationResult with test statistics
    """
    x_arr = validate_array(x)
    y_arr = validate_array(y)

    coefficient, p_value = stats.spearmanr(x_arr, y_arr)

    return CorrelationResult(
        coefficient=float(coefficient),
        p_value=float(p_value),
        test_name="Spearman",
        is_significant=p_value < 0.05
    )


def kendall_test(x: np.ndarray, y: np.ndarray) -> CorrelationResult:
    """
    Kendall tau correlation coefficient test.

    Args:
        x: First variable
        y: Second variable

    Returns:
        CorrelationResult with test statistics
    """
    x_arr = validate_array(x)
    y_arr = validate_array(y)

    coefficient, p_value = stats.kendalltau(x_arr, y_arr)

    return CorrelationResult(
        coefficient=float(coefficient),
        p_value=float(p_value),
        test_name="Kendall",
        is_significant=p_value < 0.05
    )
