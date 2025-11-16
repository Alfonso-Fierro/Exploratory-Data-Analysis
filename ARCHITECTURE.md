# EDA Suite Architecture Documentation

## Design Philosophy

The EDA Suite follows **CLEAN** and **SOLID** principles to ensure maintainability, extensibility, and professional-grade code quality.

### CLEAN Code Principles

1. **Clear**: Code is self-documenting with descriptive names
2. **Logical**: Functions have single, well-defined purposes
3. **Efficient**: Optimized for performance
4. **Accessible**: Easy to use with consistent interfaces
5. **Navigable**: Well-organized module structure

### SOLID Principles

1. **Single Responsibility**: Each function/class has one reason to change
2. **Open/Closed**: Open for extension, closed for modification
3. **Liskov Substitution**: Consistent behavior across implementations
4. **Interface Segregation**: Focused, minimal interfaces
5. **Dependency Inversion**: Depend on abstractions, not concretions

## Code Quality Standards

### Function Design
- **Maximum 40 lines per function**: Ensures readability and maintainability
- **Maximum 2 parameters**: Complex inputs use dataclasses/configuration objects
- **Type hints**: All functions have complete type annotations
- **Comprehensive docstrings**: Every function documents purpose, args, and returns

### Example

```python
@dataclass
class TestConfig:
    """Configuration for statistical tests."""
    alpha: float = 0.05
    alternative: str = "two-sided"

def shapiro_test(data: np.ndarray) -> TestResult:
    """
    Shapiro-Wilk test for normality.

    Args:
        data: Array of sample data

    Returns:
        TestResult with test statistics
    """
    arr = validate_array(data)
    statistic, p_value = stats.shapiro(arr)

    return TestResult(
        statistic=float(statistic),
        p_value=float(p_value),
        test_name="Shapiro-Wilk",
        is_significant=p_value < 0.05
    )
```

## Module Structure

### 1. Statistical Tests (`statistical_tests/`)
Tests for distribution properties and relationships.

- `normality.py`: Shapiro-Wilk, Anderson-Darling, Jarque-Bera, KS tests
- `variance.py`: Levene, Bartlett, Fligner-Killeen tests
- `correlation.py`: Pearson, Spearman, Kendall tests

### 2. Hypothesis Testing (`hypothesis_testing/`)
Parametric and non-parametric hypothesis tests.

- `parametric.py`: T-tests (one-sample, two-sample, paired)
- `anova.py`: One-way and two-way ANOVA
- `nonparametric.py`: Mann-Whitney, Wilcoxon, Kruskal-Wallis, Friedman
- `categorical.py`: Chi-square, Fisher's Exact, McNemar

### 3. Missing Data (`missing_data/`)
Analysis and detection of missing data patterns.

- `detection.py`: Missing data summaries and pattern detection
- `mechanisms.py`: MCAR/MAR/MNAR testing

### 4. Imputation (`imputation/`)
Multiple strategies for handling missing data.

- `simple.py`: Mean, median, mode imputation
- `advanced.py`: KNN, iterative, MICE imputation

### 5. Discriminant Analysis (`discriminant/`)
Classification based on Bayes' theorem.

- `lda.py`: Linear and Quadratic Discriminant Analysis

### 6. Factorial Analysis (`factorial/`)
Dimension reduction and latent variable analysis.

- `factor_analysis.py`: Factor Analysis and PCA

### 7. Time Series (`timeseries/`)
Temporal data analysis tools.

- `decomposition.py`: Seasonal decomposition, trend analysis
- `stationarity.py`: ADF and KPSS tests
- `autocorrelation.py`: ACF and PACF computation

### 8. Univariate Analysis (`univariate/`)
Single variable statistical analysis.

- `descriptive.py`: Central tendency, dispersion, moments
- `distribution.py`: Distribution fitting and testing

### 9. Bivariate Analysis (`bivariate/`)
Two-variable relationship analysis.

- `association.py`: Correlation, covariance, contingency tables
- `regression.py`: Simple linear regression

### 10. Multivariate Analysis (`multivariate/`)
Multiple variable analysis.

- `clustering.py`: K-means, hierarchical clustering
- `outliers.py`: Mahalanobis distance, multivariate outlier detection

### 11. Visualization (`visualization/`)
Statistical plotting utilities.

- `distributions.py`: Histograms, Q-Q plots, boxplots
- `relationships.py`: Scatter plots, correlation heatmaps
- `timeseries.py`: Time series plots, ACF/PACF plots

### 12. Utilities (`utils/`)
Helper functions and configurations.

- `config.py`: Configuration dataclasses
- `validators.py`: Input validation functions
- `transformers.py`: Data transformation utilities

## Data Flow

```
User Input
    ↓
Validation (utils.validators)
    ↓
Processing (module-specific functions)
    ↓
Result Dataclass
    ↓
User Output
```

## Extension Points

### Adding New Tests
1. Create function in appropriate module
2. Follow function constraints (≤40 lines, ≤2 params)
3. Return dataclass result
4. Add to module `__init__.py`
5. Document in README

### Adding New Modules
1. Create directory under `eda_suite/`
2. Add `__init__.py` with exports
3. Follow existing patterns
4. Update main `__init__.py`
5. Add examples and documentation

## Performance Considerations

- **Vectorization**: Use NumPy/Pandas operations over loops
- **Memory efficiency**: Process data in chunks when needed
- **Lazy evaluation**: Compute only when needed
- **Caching**: Store expensive computations

## Testing Strategy

Tests should cover:
1. **Unit tests**: Individual function correctness
2. **Integration tests**: Module interactions
3. **Edge cases**: Empty data, single values, extreme values
4. **Validation**: Proper error handling

## Documentation Standards

Every module/function must have:
1. **Module docstring**: Purpose and contents
2. **Function docstring**: Purpose, args, returns, examples
3. **Type hints**: Complete type annotations
4. **Examples**: Usage demonstrations

## Version Control

- Use semantic versioning (MAJOR.MINOR.PATCH)
- Maintain CHANGELOG.md
- Tag releases
- Document breaking changes

## Future Enhancements

Potential additions:
- Bayesian statistics module
- Robust statistics module
- Bootstrap and permutation tests
- Advanced visualization (interactive plots)
- Parallel processing for large datasets
- GPU acceleration options

## References

This library implements methods from:
- Montgomery, D. C. (2017). Design and Analysis of Experiments
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning
- Little, R. J., & Rubin, D. B. (2019). Statistical Analysis with Missing Data
- Box, G. E., Jenkins, G. M., & Reinsel, G. C. (2015). Time Series Analysis

## Contact

For architectural questions or contributions:
- GitHub Issues: https://github.com/Alfonso-Fierro/Exploratory-Data-Analysis/issues
- Pull Requests: Welcome with detailed descriptions
