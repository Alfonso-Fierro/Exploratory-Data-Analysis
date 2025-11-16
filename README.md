# EDA Suite: Comprehensive Exploratory Data Analysis Library

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A professional-grade Python library for comprehensive exploratory data analysis, built with CLEAN and SOLID principles. Features automated statistical analysis, hypothesis testing, missing data handling, imputation, discriminant analysis, factorial analysis, and time series analysis at a Master's level.

## Features

### 📊 Statistical Testing
- **Normality Tests**: Shapiro-Wilk, Anderson-Darling, Jarque-Bera, Kolmogorov-Smirnov
- **Variance Tests**: Levene, Bartlett, Fligner-Killeen
- **Correlation Tests**: Pearson, Spearman, Kendall

### 🔬 Hypothesis Testing
- **Parametric Tests**: One-sample t-test, Two-sample t-test, Paired t-test
- **ANOVA**: One-way ANOVA, Two-way ANOVA
- **Non-parametric Tests**: Mann-Whitney U, Wilcoxon, Kruskal-Wallis, Friedman
- **Categorical Tests**: Chi-square, Fisher's Exact, McNemar

### 🔍 Missing Data Analysis
- Missing data pattern detection
- MCAR/MAR/MNAR mechanism testing
- Comprehensive missing data visualization

### 🛠️ Data Imputation
- **Simple Methods**: Mean, Median, Mode imputation
- **Advanced Methods**: KNN imputation, Iterative imputation, MICE

### 📈 Discriminant Analysis
- Linear Discriminant Analysis (LDA)
- Quadratic Discriminant Analysis (QDA)

### 🔄 Factorial Analysis
- Factor Analysis
- Principal Component Analysis (PCA)

### ⏰ Time Series Analysis
- Seasonal decomposition
- Stationarity testing (ADF, KPSS)
- Autocorrelation (ACF/PACF)

### 📉 Univariate Analysis
- Comprehensive descriptive statistics
- Distribution fitting and testing
- Higher moment analysis

### 🔗 Bivariate Analysis
- Correlation and covariance analysis
- Simple linear regression
- Contingency tables

### 🌐 Multivariate Analysis
- K-means clustering
- Hierarchical clustering
- Multivariate outlier detection (Mahalanobis distance)

### 📊 Visualization
- Distribution plots (histograms, Q-Q plots, boxplots)
- Relationship plots (scatter, correlation heatmaps)
- Time series plots (ACF/PACF)

## Installation

```bash
pip install -e .
```

Or install from requirements:

```bash
pip install -r requirements.txt
```

## Quick Start

### Statistical Testing

```python
import numpy as np
from eda_suite.statistical_tests import shapiro_test, pearson_test

# Test for normality
data = np.random.normal(0, 1, 100)
result = shapiro_test(data)
print(f"Shapiro-Wilk: statistic={result.statistic:.4f}, p-value={result.p_value:.4f}")

# Test correlation
x = np.random.randn(100)
y = x + np.random.randn(100) * 0.5
corr_result = pearson_test(x, y)
print(f"Pearson: r={corr_result.coefficient:.4f}, p-value={corr_result.p_value:.4f}")
```

### Hypothesis Testing

```python
from eda_suite.hypothesis_testing import two_sample_ttest, one_way_anova

# Two-sample t-test
group1 = np.random.normal(0, 1, 50)
group2 = np.random.normal(0.5, 1, 50)
result = two_sample_ttest(group1, group2)
print(f"T-test: t={result.statistic:.4f}, p-value={result.p_value:.4f}")

# ANOVA
groups = [np.random.normal(i, 1, 30) for i in range(3)]
anova_result = one_way_anova(groups)
print(f"ANOVA: F={anova_result.f_statistic:.4f}, p-value={anova_result.p_value:.4f}")
```

### Missing Data Analysis and Imputation

```python
import pandas as pd
from eda_suite.missing_data import missing_summary, test_mcar
from eda_suite.imputation import mean_imputation, knn_imputation

# Create data with missing values
df = pd.DataFrame({
    'A': [1, 2, np.nan, 4, 5],
    'B': [np.nan, 2, 3, 4, 5],
    'C': [1, 2, 3, np.nan, 5]
})

# Analyze missing data
summary = missing_summary(df)
print(f"Total missing: {summary.total_missing}")
print(f"Percent missing: {summary.percent_missing:.2f}%")

# Test MCAR
mcar_result = test_mcar(df)
print(f"Is MCAR: {mcar_result.is_mcar}")

# Impute missing values
df_imputed = mean_imputation(df)
print("Imputed data:")
print(df_imputed)
```

### Discriminant Analysis

```python
from eda_suite.discriminant import LinearDiscriminantAnalysis
from sklearn.datasets import load_iris

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Perform LDA
lda = LinearDiscriminantAnalysis()
lda.fit(X, y)
X_transformed = lda.transform(X)

# Get results
results = lda.get_results()
print(f"Explained variance ratio: {results.explained_variance_ratio}")
```

### Time Series Analysis

```python
import pandas as pd
from eda_suite.timeseries import (
    seasonal_decomposition,
    adf_test,
    compute_acf
)

# Create time series
dates = pd.date_range('2020-01-01', periods=100)
ts = pd.Series(np.cumsum(np.random.randn(100)), index=dates)

# Test stationarity
adf_result = adf_test(ts)
print(f"ADF test: statistic={adf_result.statistic:.4f}")
print(f"Is stationary: {adf_result.is_stationary}")

# Compute ACF
acf_result = compute_acf(ts, nlags=20)
print(f"ACF values: {acf_result.values[:5]}")
```

### Univariate Analysis

```python
from eda_suite.univariate import compute_statistics, compute_moments, fit_distribution

data = np.random.normal(100, 15, 1000)

# Descriptive statistics
stats = compute_statistics(data)
print(f"Mean: {stats.mean:.2f}")
print(f"Std: {stats.std:.2f}")
print(f"IQR: {stats.iqr:.2f}")

# Higher moments
moments = compute_moments(data)
print(f"Skewness: {moments.skewness:.4f}")
print(f"Kurtosis: {moments.kurtosis:.4f}")

# Fit distribution
fit_result = fit_distribution(data, "norm")
print(f"Distribution fit p-value: {fit_result.p_value:.4f}")
```

### Multivariate Analysis

```python
from eda_suite.multivariate import kmeans_analysis, detect_multivariate_outliers

# Generate multivariate data
X = np.random.randn(100, 4)

# Clustering
cluster_result = kmeans_analysis(X, n_clusters=3)
print(f"Cluster labels: {cluster_result.labels[:10]}")
print(f"Inertia: {cluster_result.inertia:.2f}")

# Outlier detection
outlier_result = detect_multivariate_outliers(X, threshold=3.0)
print(f"Number of outliers: {outlier_result['n_outliers']}")
print(f"Outlier percentage: {outlier_result['outlier_percentage']:.2f}%")
```

### Visualization

```python
from eda_suite.visualization import (
    plot_histogram,
    plot_correlation_heatmap,
    plot_scatter
)
import matplotlib.pyplot as plt

# Histogram
data = np.random.normal(0, 1, 1000)
fig = plot_histogram(data)
plt.savefig('histogram.png')
plt.close()

# Correlation heatmap
df = pd.DataFrame(np.random.randn(100, 5), columns=['A', 'B', 'C', 'D', 'E'])
fig = plot_correlation_heatmap(df)
plt.savefig('correlation_heatmap.png')
plt.close()

# Scatter plot
x = np.random.randn(100)
y = 2 * x + np.random.randn(100)
fig = plot_scatter(x, y)
plt.savefig('scatter.png')
plt.close()
```

## Architecture

The library follows CLEAN and SOLID principles:

- **Single Responsibility**: Each function has one clear purpose
- **Open/Closed**: Extensible through configuration classes
- **Liskov Substitution**: Consistent interfaces across modules
- **Interface Segregation**: Modular design with focused interfaces
- **Dependency Inversion**: Abstractions over concrete implementations

### Code Quality Standards

- Functions ≤ 40 lines
- Functions accept ≤ 2 parameters (complex inputs via dataclasses)
- Self-documenting code with comprehensive docstrings
- Type hints throughout
- Performance-optimized Python

## Project Structure

```
eda_suite/
├── __init__.py
├── statistical_tests/      # Normality, variance, correlation tests
├── hypothesis_testing/     # Parametric and non-parametric tests
├── missing_data/           # Missing data analysis
├── imputation/             # Data imputation strategies
├── discriminant/           # LDA and QDA
├── factorial/              # Factor analysis and PCA
├── timeseries/             # Time series analysis
├── univariate/             # Single variable analysis
├── bivariate/              # Two variable analysis
├── multivariate/           # Multiple variable analysis
├── visualization/          # Plotting utilities
└── utils/                  # Helper functions and configurations
```

## Requirements

- Python >= 3.8
- numpy >= 1.24.0
- pandas >= 2.0.0
- scipy >= 1.10.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- statsmodels >= 0.14.0
- scikit-learn >= 1.3.0

## Contributing

Contributions are welcome! Please ensure:
- Code follows CLEAN and SOLID principles
- Functions are ≤ 40 lines with ≤ 2 parameters
- Comprehensive docstrings
- Unit tests for new features

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{eda_suite,
  title = {EDA Suite: Comprehensive Exploratory Data Analysis Library},
  author = {EDA Suite Contributors},
  year = {2024},
  url = {https://github.com/Alfonso-Fierro/Exploratory-Data-Analysis}
}
```

## Support

For issues and questions:
- GitHub Issues: https://github.com/Alfonso-Fierro/Exploratory-Data-Analysis/issues
- Documentation: See docstrings in code

## Acknowledgments

Built with modern software engineering practices and statistical rigor for the data science community.
