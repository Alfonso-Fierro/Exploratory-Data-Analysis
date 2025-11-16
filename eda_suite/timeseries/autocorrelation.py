"""
Autocorrelation analysis for time series.

Computes ACF and PACF for identifying patterns.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf, pacf
from dataclasses import dataclass


@dataclass
class AutocorrelationResult:
    """Result of autocorrelation analysis."""

    values: np.ndarray
    confidence_interval: np.ndarray
    lags: np.ndarray


def compute_acf(
    data: pd.Series,
    nlags: int = 40
) -> AutocorrelationResult:
    """
    Compute Autocorrelation Function.

    Args:
        data: Time series data
        nlags: Number of lags

    Returns:
        AutocorrelationResult with ACF values
    """
    acf_values = acf(data.dropna(), nlags=nlags)
    lags = np.arange(len(acf_values))

    conf_int = 1.96 / np.sqrt(len(data))
    confidence = np.array([conf_int, -conf_int])

    return AutocorrelationResult(
        values=acf_values,
        confidence_interval=confidence,
        lags=lags
    )


def compute_pacf(
    data: pd.Series,
    nlags: int = 40
) -> AutocorrelationResult:
    """
    Compute Partial Autocorrelation Function.

    Args:
        data: Time series data
        nlags: Number of lags

    Returns:
        AutocorrelationResult with PACF values
    """
    pacf_values = pacf(data.dropna(), nlags=nlags)
    lags = np.arange(len(pacf_values))

    conf_int = 1.96 / np.sqrt(len(data))
    confidence = np.array([conf_int, -conf_int])

    return AutocorrelationResult(
        values=pacf_values,
        confidence_interval=confidence,
        lags=lags
    )
