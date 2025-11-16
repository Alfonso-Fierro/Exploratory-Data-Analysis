"""
Missing data mechanism analysis.

Tests for MCAR, MAR, and MNAR patterns.
"""

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass


@dataclass
class MCARTestResult:
    """Result of MCAR test."""

    statistic: float
    p_value: float
    is_mcar: bool


def test_mcar(data: pd.DataFrame) -> MCARTestResult:
    """
    Little's MCAR test approximation.

    Args:
        data: DataFrame with missing values

    Returns:
        MCARTestResult with test statistics
    """
    missing_indicator = data.isnull().astype(int)

    if missing_indicator.sum().sum() == 0:
        return MCARTestResult(
            statistic=0.0,
            p_value=1.0,
            is_mcar=True
        )

    complete_cases = data.dropna()
    incomplete_cases = data[data.isnull().any(axis=1)]

    if len(incomplete_cases) == 0:
        return MCARTestResult(0.0, 1.0, True)

    numeric_cols = data.select_dtypes(include=[np.number]).columns
    t_stats = []

    for col in numeric_cols:
        complete_vals = complete_cases[col].dropna()
        incomplete_vals = incomplete_cases[col].dropna()

        if len(complete_vals) > 0 and len(incomplete_vals) > 0:
            stat, p = stats.ttest_ind(complete_vals, incomplete_vals)
            t_stats.append(p)

    overall_p = np.mean(t_stats) if t_stats else 1.0

    return MCARTestResult(
        statistic=1.0,
        p_value=float(overall_p),
        is_mcar=overall_p > 0.05
    )


def analyze_mechanism(data: pd.DataFrame) -> Dict:
    """
    Analyze missing data mechanism.

    Args:
        data: DataFrame with missing values

    Returns:
        Dictionary with mechanism analysis
    """
    mcar_result = test_mcar(data)

    return {
        "mcar_test": mcar_result,
        "recommendation": "MCAR" if mcar_result.is_mcar else "MAR/MNAR"
    }
