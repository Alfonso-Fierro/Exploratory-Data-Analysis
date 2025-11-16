"""
Univariate analysis module for EDA Suite.

Provides comprehensive single-variable statistical analysis.
"""

from eda_suite.univariate.descriptive import (
    compute_statistics,
    compute_moments,
    compute_quantiles
)
from eda_suite.univariate.distribution import (
    fit_distribution,
    test_distribution_fit
)

__all__ = [
    "compute_statistics",
    "compute_moments",
    "compute_quantiles",
    "fit_distribution",
    "test_distribution_fit"
]
