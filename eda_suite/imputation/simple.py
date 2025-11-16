"""
Simple imputation methods.

Basic statistical imputation strategies.
"""

import numpy as np
import pandas as pd
from typing import Union


def mean_imputation(data: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values with column means.

    Args:
        data: DataFrame with missing values

    Returns:
        DataFrame with imputed values
    """
    result = data.copy()
    numeric_cols = result.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        mean_val = result[col].mean()
        result[col].fillna(mean_val, inplace=True)

    return result


def median_imputation(data: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values with column medians.

    Args:
        data: DataFrame with missing values

    Returns:
        DataFrame with imputed values
    """
    result = data.copy()
    numeric_cols = result.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        median_val = result[col].median()
        result[col].fillna(median_val, inplace=True)

    return result


def mode_imputation(data: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values with column modes.

    Args:
        data: DataFrame with missing values

    Returns:
        DataFrame with imputed values
    """
    result = data.copy()

    for col in result.columns:
        if result[col].isnull().any():
            mode_val = result[col].mode()
            if len(mode_val) > 0:
                result[col].fillna(mode_val[0], inplace=True)

    return result
