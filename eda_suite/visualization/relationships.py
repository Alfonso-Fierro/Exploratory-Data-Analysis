"""
Relationship visualization functions.

Plots for bivariate and multivariate analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from eda_suite.utils.config import VisualizationConfig


def plot_scatter(
    x: np.ndarray,
    y: np.ndarray,
    config: VisualizationConfig = VisualizationConfig()
) -> plt.Figure:
    """
    Create scatter plot with regression line.

    Args:
        x: X-axis data
        y: Y-axis data
        config: Visualization configuration

    Returns:
        Matplotlib figure
    """
    sns.set_style(config.style)
    fig, ax = plt.subplots(figsize=config.figsize)

    ax.scatter(x, y, alpha=0.6)

    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    ax.plot(x, p(x), "r--", alpha=0.8)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Scatter Plot with Regression Line')

    return fig


def plot_correlation_heatmap(
    data: pd.DataFrame,
    config: VisualizationConfig = VisualizationConfig()
) -> plt.Figure:
    """
    Create correlation heatmap.

    Args:
        data: Input DataFrame
        config: Visualization configuration

    Returns:
        Matplotlib figure
    """
    sns.set_style(config.style)
    fig, ax = plt.subplots(figsize=config.figsize)

    corr = data.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Correlation Heatmap')

    return fig
