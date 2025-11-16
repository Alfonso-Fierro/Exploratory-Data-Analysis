"""
Stationarity tests for time series.

Tests whether time series has constant statistical properties.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from dataclasses import dataclass


@dataclass
class StationarityResult:
    """Result of stationarity test."""

    statistic: float
    p_value: float
    is_stationary: bool
    test_name: str


def adf_test(data: pd.Series) -> StationarityResult:
    """
    Augmented Dickey-Fuller test for stationarity.

    Args:
        data: Time series data

    Returns:
        StationarityResult with test statistics
    """
    result = adfuller(data.dropna())

    return StationarityResult(
        statistic=float(result[0]),
        p_value=float(result[1]),
        is_stationary=result[1] < 0.05,
        test_name="Augmented Dickey-Fuller"
    )


def kpss_test(data: pd.Series) -> StationarityResult:
    """
    KPSS test for stationarity.

    Args:
        data: Time series data

    Returns:
        StationarityResult with test statistics
    """
    result = kpss(data.dropna(), regression='c')

    return StationarityResult(
        statistic=float(result[0]),
        p_value=float(result[1]),
        is_stationary=result[1] > 0.05,
        test_name="KPSS"
    )


def test_stationarity(data: pd.Series) -> dict:
    """
    Comprehensive stationarity testing.

    Args:
        data: Time series data

    Returns:
        Dictionary with multiple test results
    """
    adf_result = adf_test(data)
    kpss_result = kpss_test(data)

    return {
        "adf": adf_result,
        "kpss": kpss_result,
        "is_stationary": adf_result.is_stationary and kpss_result.is_stationary
    }
