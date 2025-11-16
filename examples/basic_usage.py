"""
Basic usage examples for EDA Suite.

Demonstrates common workflows for exploratory data analysis.
"""

import numpy as np
import pandas as pd
from eda_suite.statistical_tests import shapiro_test, pearson_test
from eda_suite.hypothesis_testing import two_sample_ttest
from eda_suite.univariate import compute_statistics, compute_moments
from eda_suite.missing_data import missing_summary
from eda_suite.imputation import mean_imputation


def example_statistical_tests():
    """Example: Statistical testing workflow."""
    print("=" * 50)
    print("Statistical Tests Example")
    print("=" * 50)

    data = np.random.normal(0, 1, 100)
    result = shapiro_test(data)

    print(f"Shapiro-Wilk Test:")
    print(f"  Statistic: {result.statistic:.4f}")
    print(f"  P-value: {result.p_value:.4f}")
    print(f"  Significant: {result.is_significant}")
    print()


def example_hypothesis_testing():
    """Example: Hypothesis testing workflow."""
    print("=" * 50)
    print("Hypothesis Testing Example")
    print("=" * 50)

    group1 = np.random.normal(0, 1, 50)
    group2 = np.random.normal(0.3, 1, 50)

    result = two_sample_ttest(group1, group2)

    print(f"Two-Sample T-Test:")
    print(f"  Statistic: {result.statistic:.4f}")
    print(f"  P-value: {result.p_value:.4f}")
    print(f"  Degrees of Freedom: {result.degrees_freedom}")
    print(f"  Reject Null: {result.reject_null}")
    print()


def example_descriptive_statistics():
    """Example: Descriptive statistics workflow."""
    print("=" * 50)
    print("Descriptive Statistics Example")
    print("=" * 50)

    data = np.random.normal(100, 15, 1000)

    stats = compute_statistics(data)
    print(f"Descriptive Statistics:")
    print(f"  Mean: {stats.mean:.2f}")
    print(f"  Median: {stats.median:.2f}")
    print(f"  Std Dev: {stats.std:.2f}")
    print(f"  Min: {stats.min:.2f}")
    print(f"  Max: {stats.max:.2f}")
    print(f"  IQR: {stats.iqr:.2f}")
    print()

    moments = compute_moments(data)
    print(f"Higher Moments:")
    print(f"  Skewness: {moments.skewness:.4f}")
    print(f"  Kurtosis: {moments.kurtosis:.4f}")
    print()


def example_missing_data():
    """Example: Missing data analysis workflow."""
    print("=" * 50)
    print("Missing Data Analysis Example")
    print("=" * 50)

    df = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5, 6, np.nan, 8, 9, 10],
        'B': [np.nan, 2, 3, 4, 5, np.nan, 7, 8, 9, 10],
        'C': [1, 2, 3, 4, 5, 6, 7, 8, np.nan, 10]
    })

    summary = missing_summary(df)
    print(f"Missing Data Summary:")
    print(f"  Total Missing: {summary.total_missing}")
    print(f"  Percent Missing: {summary.percent_missing:.2f}%")
    print(f"\nMissing by Column:")
    print(summary.missing_by_column)
    print()

    df_imputed = mean_imputation(df)
    print(f"Imputed DataFrame:")
    print(df_imputed)
    print()


if __name__ == "__main__":
    example_statistical_tests()
    example_hypothesis_testing()
    example_descriptive_statistics()
    example_missing_data()
