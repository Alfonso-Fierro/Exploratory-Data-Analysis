"""
Simple regression analysis.

Linear regression for bivariate relationships.
"""

import numpy as np
from scipy import stats
from dataclasses import dataclass
from eda_suite.utils.validators import validate_array


@dataclass
class RegressionResult:
    """Result of simple linear regression."""

    slope: float
    intercept: float
    r_squared: float
    p_value: float
    std_error: float


def simple_linear_regression(
    x: np.ndarray,
    y: np.ndarray
) -> RegressionResult:
    """
    Perform simple linear regression.

    Args:
        x: Independent variable
        y: Dependent variable

    Returns:
        RegressionResult with regression statistics
    """
    x_arr = validate_array(x)
    y_arr = validate_array(y)

    slope, intercept, r_value, p_value, std_err = stats.linregress(x_arr, y_arr)

    return RegressionResult(
        slope=float(slope),
        intercept=float(intercept),
        r_squared=float(r_value ** 2),
        p_value=float(p_value),
        std_error=float(std_err)
    )


def compute_r_squared(
    x: np.ndarray,
    y: np.ndarray
) -> float:
    """
    Compute R-squared (coefficient of determination).

    Args:
        x: Independent variable
        y: Dependent variable

    Returns:
        R-squared value
    """
    result = simple_linear_regression(x, y)
    return result.r_squared
