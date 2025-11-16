"""
Missing data detection and analysis.

Identifies patterns and characteristics of missing data.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict


@dataclass
class MissingSummary:
    """Summary statistics for missing data."""

    total_missing: int
    percent_missing: float
    missing_by_column: pd.Series
    missing_by_row: pd.Series


def missing_summary(data: pd.DataFrame) -> MissingSummary:
    """
    Generate comprehensive missing data summary.

    Args:
        data: Input DataFrame

    Returns:
        MissingSummary with statistics
    """
    total = data.isnull().sum().sum()
    total_cells = data.size
    percent = (total / total_cells) * 100

    by_column = data.isnull().sum()
    by_row = data.isnull().sum(axis=1)

    return MissingSummary(
        total_missing=int(total),
        percent_missing=float(percent),
        missing_by_column=by_column,
        missing_by_row=by_row
    )


def missing_heatmap_data(data: pd.DataFrame) -> np.ndarray:
    """
    Create binary matrix for missing data visualization.

    Args:
        data: Input DataFrame

    Returns:
        Binary array where 1 indicates missing
    """
    return data.isnull().astype(int).values


def missing_patterns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Identify unique missing data patterns.

    Args:
        data: Input DataFrame

    Returns:
        DataFrame with pattern counts
    """
    missing_matrix = data.isnull()
    patterns = missing_matrix.groupby(
        list(missing_matrix.columns)
    ).size().reset_index(name='count')

    return patterns.sort_values('count', ascending=False)
