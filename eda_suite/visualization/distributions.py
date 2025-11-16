"""
Distribution visualization functions.

Plots for univariate data exploration.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from eda_suite.utils.config import VisualizationConfig


def plot_histogram(
    data: np.ndarray,
    config: VisualizationConfig = VisualizationConfig()
) -> plt.Figure:
    """
    Create histogram with density overlay.

    Args:
        data: Input array
        config: Visualization configuration

    Returns:
        Matplotlib figure
    """
    sns.set_style(config.style)
    fig, ax = plt.subplots(figsize=config.figsize)

    ax.hist(data, bins='auto', density=True, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.set_title('Distribution Histogram')

    return fig


def plot_qq(
    data: np.ndarray,
    config: VisualizationConfig = VisualizationConfig()
) -> plt.Figure:
    """
    Create Q-Q plot for normality assessment.

    Args:
        data: Input array
        config: Visualization configuration

    Returns:
        Matplotlib figure
    """
    sns.set_style(config.style)
    fig, ax = plt.subplots(figsize=config.figsize)

    stats.probplot(data, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot')

    return fig


def plot_boxplot(
    data: pd.DataFrame,
    config: VisualizationConfig = VisualizationConfig()
) -> plt.Figure:
    """
    Create boxplot for outlier detection.

    Args:
        data: Input DataFrame
        config: Visualization configuration

    Returns:
        Matplotlib figure
    """
    sns.set_style(config.style)
    fig, ax = plt.subplots(figsize=config.figsize)

    data.boxplot(ax=ax)
    ax.set_title('Boxplot')

    return fig
