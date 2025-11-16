"""
Time series decomposition methods.

Separates trend, seasonal, and residual components.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from dataclasses import dataclass
from typing import Optional


@dataclass
class DecompositionResult:
    """Result of time series decomposition."""

    trend: pd.Series
    seasonal: pd.Series
    residual: pd.Series
    observed: pd.Series


def seasonal_decomposition(
    data: pd.Series,
    period: int
) -> DecompositionResult:
    """
    Decompose time series into components.

    Args:
        data: Time series data
        period: Seasonal period

    Returns:
        DecompositionResult with components
    """
    result = seasonal_decompose(
        data,
        model='additive',
        period=period
    )

    return DecompositionResult(
        trend=result.trend,
        seasonal=result.seasonal,
        residual=result.resid,
        observed=result.observed
    )


def trend_analysis(data: pd.Series) -> pd.Series:
    """
    Extract trend component using moving average.

    Args:
        data: Time series data

    Returns:
        Trend series
    """
    window_size = min(len(data) // 4, 12)
    return data.rolling(window=window_size, center=True).mean()
