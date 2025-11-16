"""
Bivariate analysis module for EDA Suite.

Provides tools for analyzing relationships between two variables.
"""

from eda_suite.bivariate.association import (
    compute_correlation_matrix,
    compute_covariance,
    compute_contingency
)
from eda_suite.bivariate.regression import (
    simple_linear_regression,
    compute_r_squared
)

__all__ = [
    "compute_correlation_matrix",
    "compute_covariance",
    "compute_contingency",
    "simple_linear_regression",
    "compute_r_squared"
]
