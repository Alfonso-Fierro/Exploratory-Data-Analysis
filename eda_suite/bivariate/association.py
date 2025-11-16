"""
Association measures for bivariate data.

Computes correlation, covariance, and contingency measures.
"""

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass


@dataclass
class CorrelationMatrix:
    """Correlation matrix with p-values."""

    correlation: np.ndarray
    p_values: np.ndarray
    variables: list


def compute_correlation_matrix(data: pd.DataFrame) -> CorrelationMatrix:
    """
    Compute correlation matrix with significance.

    Args:
        data: DataFrame with numeric variables

    Returns:
        CorrelationMatrix with correlations and p-values
    """
    numeric_data = data.select_dtypes(include=[np.number])
    n_vars = len(numeric_data.columns)

    corr_matrix = np.zeros((n_vars, n_vars))
    p_matrix = np.zeros((n_vars, n_vars))

    for i, col1 in enumerate(numeric_data.columns):
        for j, col2 in enumerate(numeric_data.columns):
            if i == j:
                corr_matrix[i, j] = 1.0
                p_matrix[i, j] = 0.0
            else:
                r, p = stats.pearsonr(
                    numeric_data[col1].dropna(),
                    numeric_data[col2].dropna()
                )
                corr_matrix[i, j] = r
                p_matrix[i, j] = p

    return CorrelationMatrix(
        correlation=corr_matrix,
        p_values=p_matrix,
        variables=list(numeric_data.columns)
    )


def compute_covariance(
    x: np.ndarray,
    y: np.ndarray
) -> float:
    """
    Compute covariance between two variables.

    Args:
        x: First variable
        y: Second variable

    Returns:
        Covariance value
    """
    return float(np.cov(x, y)[0, 1])


def compute_contingency(
    x: pd.Series,
    y: pd.Series
) -> pd.DataFrame:
    """
    Compute contingency table for categorical variables.

    Args:
        x: First categorical variable
        y: Second categorical variable

    Returns:
        Contingency table DataFrame
    """
    return pd.crosstab(x, y)
