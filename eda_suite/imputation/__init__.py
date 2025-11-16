"""
Data imputation module for EDA Suite.

Provides multiple imputation strategies for missing data.
"""

from eda_suite.imputation.simple import (
    mean_imputation,
    median_imputation,
    mode_imputation
)
from eda_suite.imputation.advanced import (
    knn_imputation,
    iterative_imputation,
    mice_imputation
)

__all__ = [
    "mean_imputation",
    "median_imputation",
    "mode_imputation",
    "knn_imputation",
    "iterative_imputation",
    "mice_imputation"
]
