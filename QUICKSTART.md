# EDA Suite Quick Start Guide

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Verify Installation

```bash
# Run the quick test
python examples/quick_test.py
```

Expected output:
```
Testing imports...
✓ All modules imported successfully
Testing statistical tests...
✓ Statistical tests working
Testing hypothesis testing...
✓ Hypothesis testing working
...
Tests passed: 6/6
```

## 5-Minute Tutorial

### 1. Statistical Testing

```python
import numpy as np
from eda_suite.statistical_tests import shapiro_test

# Test for normality
data = np.random.normal(0, 1, 100)
result = shapiro_test(data)
print(f"Normality test p-value: {result.p_value:.4f}")
```

### 2. Hypothesis Testing

```python
from eda_suite.hypothesis_testing import two_sample_ttest

# Compare two groups
group1 = np.random.normal(0, 1, 50)
group2 = np.random.normal(0.5, 1, 50)
result = two_sample_ttest(group1, group2)
print(f"Significant difference: {result.reject_null}")
```

### 3. Missing Data Handling

```python
import pandas as pd
from eda_suite.missing_data import missing_summary
from eda_suite.imputation import mean_imputation

# Analyze missing data
df = pd.DataFrame({
    'A': [1, 2, np.nan, 4, 5],
    'B': [1, 2, 3, 4, 5]
})

summary = missing_summary(df)
print(f"Missing: {summary.percent_missing:.1f}%")

# Impute missing values
df_clean = mean_imputation(df)
```

### 4. Univariate Analysis

```python
from eda_suite.univariate import compute_statistics

data = np.random.normal(100, 15, 1000)
stats = compute_statistics(data)
print(f"Mean: {stats.mean:.2f}, Std: {stats.std:.2f}")
```

### 5. Multivariate Analysis

```python
from eda_suite.multivariate import kmeans_analysis

# Cluster data
X = np.random.randn(100, 4)
result = kmeans_analysis(X, n_clusters=3)
print(f"Clusters assigned: {len(set(result.labels))}")
```

## Common Workflows

### Complete EDA Workflow

```python
import numpy as np
import pandas as pd
from eda_suite.univariate import compute_statistics, compute_moments
from eda_suite.statistical_tests import shapiro_test
from eda_suite.visualization import plot_histogram
import matplotlib.pyplot as plt

# Generate data
data = np.random.normal(100, 15, 1000)

# 1. Descriptive statistics
stats = compute_statistics(data)
print(f"Mean: {stats.mean:.2f}")
print(f"Median: {stats.median:.2f}")
print(f"Std Dev: {stats.std:.2f}")

# 2. Test normality
norm_test = shapiro_test(data)
print(f"\nNormality test p-value: {norm_test.p_value:.4f}")

# 3. Higher moments
moments = compute_moments(data)
print(f"Skewness: {moments.skewness:.4f}")
print(f"Kurtosis: {moments.kurtosis:.4f}")

# 4. Visualize
fig = plot_histogram(data)
plt.savefig('distribution.png')
plt.close()
```

### Time Series Analysis Workflow

```python
import pandas as pd
from eda_suite.timeseries import (
    adf_test,
    seasonal_decomposition,
    compute_acf
)

# Create time series
dates = pd.date_range('2020-01-01', periods=365)
ts = pd.Series(np.cumsum(np.random.randn(365)), index=dates)

# Test stationarity
adf_result = adf_test(ts)
print(f"Is stationary: {adf_result.is_stationary}")

# Compute autocorrelation
acf_result = compute_acf(ts, nlags=20)
print(f"ACF at lag 1: {acf_result.values[1]:.4f}")
```

### Missing Data Workflow

```python
import pandas as pd
import numpy as np
from eda_suite.missing_data import missing_summary, test_mcar
from eda_suite.imputation import knn_imputation, ImputationConfig

# Data with missing values
df = pd.DataFrame({
    'A': [1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10],
    'B': [1, 2, 3, np.nan, 5, 6, 7, np.nan, 9, 10],
    'C': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
})

# 1. Analyze missing patterns
summary = missing_summary(df)
print(f"Total missing: {summary.total_missing}")
print(summary.missing_by_column)

# 2. Test MCAR
mcar = test_mcar(df)
print(f"\nIs MCAR: {mcar.is_mcar}")

# 3. Impute
config = ImputationConfig(strategy='knn', n_neighbors=3)
df_imputed = knn_imputation(df, config)
print(f"\nMissing after imputation: {df_imputed.isnull().sum().sum()}")
```

## Module Guide

| Module | Use Case |
|--------|----------|
| `statistical_tests` | Test distributions and relationships |
| `hypothesis_testing` | Compare groups and test hypotheses |
| `missing_data` | Analyze missing data patterns |
| `imputation` | Fill missing values |
| `discriminant` | Classification and dimension reduction |
| `factorial` | Factor analysis and PCA |
| `timeseries` | Analyze temporal data |
| `univariate` | Single variable analysis |
| `bivariate` | Two variable relationships |
| `multivariate` | Multiple variable analysis |
| `visualization` | Create plots |

## Configuration

Many functions accept configuration objects:

```python
from eda_suite.utils.config import TestConfig, VisualizationConfig

# Statistical test configuration
test_config = TestConfig(alpha=0.01, alternative='two-sided')

# Visualization configuration
viz_config = VisualizationConfig(
    figsize=(12, 6),
    style='darkgrid',
    palette='muted'
)
```

## Next Steps

1. **Explore Examples**: Run `examples/basic_usage.py` and `examples/advanced_analysis.py`
2. **Read Documentation**: Check `README.md` for comprehensive examples
3. **Architecture**: See `ARCHITECTURE.md` for design details
4. **Customize**: Extend modules following CLEAN and SOLID principles

## Getting Help

- Check function docstrings: `help(function_name)`
- Read module documentation in code
- Review examples in `examples/` directory
- See `README.md` for detailed usage

## Tips

1. **Import Patterns**: Import specific functions for cleaner code
   ```python
   from eda_suite.statistical_tests import shapiro_test
   ```

2. **Configuration**: Use config objects for complex parameters
   ```python
   config = ImputationConfig(strategy='knn', n_neighbors=5)
   result = knn_imputation(data, config)
   ```

3. **Results**: All tests return structured dataclass results
   ```python
   result = shapiro_test(data)
   print(result.statistic, result.p_value)
   ```

4. **Chain Operations**: Results are designed for easy chaining
   ```python
   data = mean_imputation(df)
   stats = compute_statistics(data['column'])
   ```

## Performance Tips

- Use NumPy arrays for numerical operations
- Process large datasets in chunks
- Cache expensive computations
- Use vectorized operations over loops

Happy Analyzing! 📊
