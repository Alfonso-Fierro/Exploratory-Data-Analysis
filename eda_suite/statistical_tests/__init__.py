"""
Statistical tests module for EDA Suite.

Provides normality, variance, and correlation tests.
"""

from eda_suite.statistical_tests.normality import (
    shapiro_test,
    anderson_test,
    jarque_bera_test,
    kolmogorov_smirnov_test
)
from eda_suite.statistical_tests.variance import (
    levene_test,
    bartlett_test,
    fligner_test
)
from eda_suite.statistical_tests.correlation import (
    pearson_test,
    spearman_test,
    kendall_test
)

__all__ = [
    "shapiro_test",
    "anderson_test",
    "jarque_bera_test",
    "kolmogorov_smirnov_test",
    "levene_test",
    "bartlett_test",
    "fligner_test",
    "pearson_test",
    "spearman_test",
    "kendall_test"
]
