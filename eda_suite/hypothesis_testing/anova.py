"""
Analysis of Variance (ANOVA) tests.

Implements one-way and two-way ANOVA.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats
from typing import List
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm


@dataclass
class ANOVAResult:
    """Result of ANOVA test."""

    f_statistic: float
    p_value: float
    degrees_freedom_between: int
    degrees_freedom_within: int
    reject_null: bool


def one_way_anova(groups: List[np.ndarray]) -> ANOVAResult:
    """
    One-way ANOVA test.

    Args:
        groups: List of sample groups

    Returns:
        ANOVAResult with test statistics
    """
    f_stat, p_value = stats.f_oneway(*groups)

    n_groups = len(groups)
    n_total = sum(len(g) for g in groups)
    df_between = n_groups - 1
    df_within = n_total - n_groups

    return ANOVAResult(
        f_statistic=float(f_stat),
        p_value=float(p_value),
        degrees_freedom_between=df_between,
        degrees_freedom_within=df_within,
        reject_null=p_value < 0.05
    )


def two_way_anova(
    data: pd.DataFrame,
    formula: str
) -> pd.DataFrame:
    """
    Two-way ANOVA test.

    Args:
        data: DataFrame with variables
        formula: R-style formula (e.g., 'y ~ C(A) + C(B) + C(A):C(B)')

    Returns:
        DataFrame with ANOVA table
    """
    model = ols(formula, data=data).fit()
    anova_table = anova_lm(model, typ=2)

    return anova_table
