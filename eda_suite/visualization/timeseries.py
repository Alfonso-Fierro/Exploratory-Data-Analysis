"""
Time series visualization functions.

Plots for temporal data analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from eda_suite.utils.config import VisualizationConfig


def plot_timeseries(
    data: pd.Series,
    config: VisualizationConfig = VisualizationConfig()
) -> plt.Figure:
    """
    Create time series plot.

    Args:
        data: Time series data
        config: Visualization configuration

    Returns:
        Matplotlib figure
    """
    sns.set_style(config.style)
    fig, ax = plt.subplots(figsize=config.figsize)

    ax.plot(data.index, data.values)
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title('Time Series Plot')
    ax.grid(True, alpha=0.3)

    return fig


def plot_acf_pacf(
    acf_values: np.ndarray,
    pacf_values: np.ndarray
) -> plt.Figure:
    """
    Create ACF and PACF plots.

    Args:
        acf_values: ACF values
        pacf_values: PACF values

    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.stem(acf_values)
    ax1.set_title('Autocorrelation Function (ACF)')
    ax1.set_xlabel('Lag')
    ax1.set_ylabel('ACF')

    ax2.stem(pacf_values)
    ax2.set_title('Partial Autocorrelation Function (PACF)')
    ax2.set_xlabel('Lag')
    ax2.set_ylabel('PACF')

    plt.tight_layout()
    return fig
