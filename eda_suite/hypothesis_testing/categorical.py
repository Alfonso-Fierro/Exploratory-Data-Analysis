"""
Categorical data hypothesis tests.

Implements chi-square and related tests for categorical data.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class CategoricalTestResult:
    """Result of categorical test."""

    statistic: float
    p_value: float
    test_name: str
    reject_null: bool


def chi_square_test(
    observed: np.ndarray,
    expected: np.ndarray = None
) -> CategoricalTestResult:
    """
    Chi-square test for independence.

    Args:
        observed: Observed frequency table
        expected: Expected frequency table (optional)

    Returns:
        CategoricalTestResult with test statistics
    """
    if expected is None:
        statistic, p_value, dof, expected = stats.chi2_contingency(observed)
    else:
        statistic, p_value = stats.chisquare(observed, expected)

    return CategoricalTestResult(
        statistic=float(statistic),
        p_value=float(p_value),
        test_name="Chi-Square",
        reject_null=p_value < 0.05
    )


def fisher_exact_test(contingency_table: np.ndarray) -> CategoricalTestResult:
    """
    Fisher's exact test for 2x2 contingency tables.

    Args:
        contingency_table: 2x2 contingency table

    Returns:
        CategoricalTestResult with test statistics
    """
    odds_ratio, p_value = stats.fisher_exact(contingency_table)

    return CategoricalTestResult(
        statistic=float(odds_ratio),
        p_value=float(p_value),
        test_name="Fisher's Exact",
        reject_null=p_value < 0.05
    )


def mcnemar_test(contingency_table: np.ndarray) -> CategoricalTestResult:
    """
    McNemar's test for paired nominal data.

    Args:
        contingency_table: 2x2 contingency table

    Returns:
        CategoricalTestResult with test statistics
    """
    from statsmodels.stats.contingency_tables import mcnemar

    result = mcnemar(contingency_table, exact=True)

    return CategoricalTestResult(
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        test_name="McNemar",
        reject_null=result.pvalue < 0.05
    )
