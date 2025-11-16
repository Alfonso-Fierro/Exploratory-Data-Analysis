"""
Time series analysis module for EDA Suite.

Provides tools for analyzing temporal data patterns.
"""

from eda_suite.timeseries.decomposition import (
    seasonal_decomposition,
    trend_analysis
)
from eda_suite.timeseries.stationarity import (
    adf_test,
    kpss_test,
    test_stationarity
)
from eda_suite.timeseries.autocorrelation import (
    compute_acf,
    compute_pacf
)

__all__ = [
    "seasonal_decomposition",
    "trend_analysis",
    "adf_test",
    "kpss_test",
    "test_stationarity",
    "compute_acf",
    "compute_pacf"
]
