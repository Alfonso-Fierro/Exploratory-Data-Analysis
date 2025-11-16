"""
Advanced imputation methods.

Sophisticated imputation strategies using ML algorithms.
"""

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, IterativeImputer
from eda_suite.utils.config import ImputationConfig


def knn_imputation(
    data: pd.DataFrame,
    config: ImputationConfig = ImputationConfig()
) -> pd.DataFrame:
    """
    K-Nearest Neighbors imputation.

    Args:
        data: DataFrame with missing values
        config: Imputation configuration

    Returns:
        DataFrame with imputed values
    """
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    result = data.copy()

    imputer = KNNImputer(n_neighbors=config.n_neighbors)
    result[numeric_cols] = imputer.fit_transform(data[numeric_cols])

    return result


def iterative_imputation(
    data: pd.DataFrame,
    config: ImputationConfig = ImputationConfig()
) -> pd.DataFrame:
    """
    Iterative multivariate imputation.

    Args:
        data: DataFrame with missing values
        config: Imputation configuration

    Returns:
        DataFrame with imputed values
    """
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    result = data.copy()

    imputer = IterativeImputer(max_iter=config.max_iter, random_state=42)
    result[numeric_cols] = imputer.fit_transform(data[numeric_cols])

    return result


def mice_imputation(
    data: pd.DataFrame,
    config: ImputationConfig = ImputationConfig()
) -> pd.DataFrame:
    """
    Multiple Imputation by Chained Equations (MICE).

    Args:
        data: DataFrame with missing values
        config: Imputation configuration

    Returns:
        DataFrame with imputed values
    """
    return iterative_imputation(data, config)
