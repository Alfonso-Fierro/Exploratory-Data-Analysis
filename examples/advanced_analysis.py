"""
Advanced analysis examples for EDA Suite.

Demonstrates complex analytical workflows.
"""

import numpy as np
import pandas as pd
from eda_suite.discriminant import LinearDiscriminantAnalysis
from eda_suite.factorial import PrincipalComponentAnalysis
from eda_suite.multivariate import kmeans_analysis, detect_multivariate_outliers
from eda_suite.timeseries import adf_test, compute_acf


def example_discriminant_analysis():
    """Example: LDA workflow."""
    print("=" * 50)
    print("Linear Discriminant Analysis Example")
    print("=" * 50)

    np.random.seed(42)
    class1 = np.random.randn(50, 4) + np.array([0, 0, 0, 0])
    class2 = np.random.randn(50, 4) + np.array([2, 2, 2, 2])
    class3 = np.random.randn(50, 4) + np.array([4, 4, 4, 4])

    X = np.vstack([class1, class2, class3])
    y = np.array([0] * 50 + [1] * 50 + [2] * 50)

    lda = LinearDiscriminantAnalysis()
    lda.fit(X, y)

    results = lda.get_results()
    print(f"LDA Results:")
    print(f"  Explained Variance Ratio: {results.explained_variance_ratio}")
    print(f"  Number of Components: {results.coefficients.shape}")
    print()


def example_pca():
    """Example: PCA workflow."""
    print("=" * 50)
    print("Principal Component Analysis Example")
    print("=" * 50)

    np.random.seed(42)
    X = np.random.randn(100, 5)

    pca = PrincipalComponentAnalysis(n_components=3)
    pca.fit(X)
    X_transformed = pca.transform(X)

    print(f"PCA Results:")
    print(f"  Original shape: {X.shape}")
    print(f"  Transformed shape: {X_transformed.shape}")
    print()


def example_clustering():
    """Example: Clustering workflow."""
    print("=" * 50)
    print("K-Means Clustering Example")
    print("=" * 50)

    np.random.seed(42)
    cluster1 = np.random.randn(30, 2) + np.array([0, 0])
    cluster2 = np.random.randn(30, 2) + np.array([5, 5])
    cluster3 = np.random.randn(30, 2) + np.array([10, 0])

    X = np.vstack([cluster1, cluster2, cluster3])

    result = kmeans_analysis(X, n_clusters=3)

    print(f"Clustering Results:")
    print(f"  Number of clusters: {result.n_clusters}")
    print(f"  Inertia: {result.inertia:.2f}")
    print(f"  Labels: {result.labels[:10]}")
    print()


def example_outlier_detection():
    """Example: Multivariate outlier detection."""
    print("=" * 50)
    print("Multivariate Outlier Detection Example")
    print("=" * 50)

    np.random.seed(42)
    normal_data = np.random.randn(95, 3)
    outliers = np.random.randn(5, 3) * 5 + 10

    X = np.vstack([normal_data, outliers])

    result = detect_multivariate_outliers(X, threshold=3.0)

    print(f"Outlier Detection Results:")
    print(f"  Number of outliers: {result['n_outliers']}")
    print(f"  Outlier percentage: {result['outlier_percentage']:.2f}%")
    print(f"  Outlier indices: {result['outlier_indices']}")
    print()


def example_time_series():
    """Example: Time series analysis."""
    print("=" * 50)
    print("Time Series Analysis Example")
    print("=" * 50)

    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=100)
    ts = pd.Series(np.cumsum(np.random.randn(100)), index=dates)

    adf_result = adf_test(ts)
    print(f"ADF Test Results:")
    print(f"  Statistic: {adf_result.statistic:.4f}")
    print(f"  P-value: {adf_result.p_value:.4f}")
    print(f"  Is Stationary: {adf_result.is_stationary}")
    print()

    acf_result = compute_acf(ts, nlags=10)
    print(f"ACF Values (first 5 lags):")
    print(acf_result.values[:5])
    print()


if __name__ == "__main__":
    example_discriminant_analysis()
    example_pca()
    example_clustering()
    example_outlier_detection()
    example_time_series()
