"""
Quick validation test for EDA Suite.

Tests that all modules can be imported and basic functionality works.
"""

import numpy as np
import pandas as pd


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        from eda_suite import statistical_tests
        from eda_suite import hypothesis_testing
        from eda_suite import missing_data
        from eda_suite import imputation
        from eda_suite import discriminant
        from eda_suite import factorial
        from eda_suite import timeseries
        from eda_suite import univariate
        from eda_suite import bivariate
        from eda_suite import multivariate
        from eda_suite import visualization
        from eda_suite import utils
        print("✓ All modules imported successfully")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_statistical_tests():
    """Test statistical tests module."""
    print("Testing statistical tests...")

    try:
        from eda_suite.statistical_tests import shapiro_test

        data = np.random.normal(0, 1, 100)
        result = shapiro_test(data)
        assert hasattr(result, 'statistic')
        assert hasattr(result, 'p_value')
        print("✓ Statistical tests working")
        return True
    except Exception as e:
        print(f"✗ Statistical tests failed: {e}")
        return False


def test_hypothesis_testing():
    """Test hypothesis testing module."""
    print("Testing hypothesis testing...")

    try:
        from eda_suite.hypothesis_testing import two_sample_ttest

        group1 = np.random.normal(0, 1, 50)
        group2 = np.random.normal(0, 1, 50)
        result = two_sample_ttest(group1, group2)
        assert hasattr(result, 'statistic')
        assert hasattr(result, 'p_value')
        print("✓ Hypothesis testing working")
        return True
    except Exception as e:
        print(f"✗ Hypothesis testing failed: {e}")
        return False


def test_missing_data():
    """Test missing data module."""
    print("Testing missing data analysis...")

    try:
        from eda_suite.missing_data import missing_summary

        df = pd.DataFrame({
            'A': [1, 2, np.nan, 4, 5],
            'B': [1, 2, 3, 4, 5]
        })
        result = missing_summary(df)
        assert hasattr(result, 'total_missing')
        print("✓ Missing data analysis working")
        return True
    except Exception as e:
        print(f"✗ Missing data analysis failed: {e}")
        return False


def test_imputation():
    """Test imputation module."""
    print("Testing imputation...")

    try:
        from eda_suite.imputation import mean_imputation

        df = pd.DataFrame({
            'A': [1, 2, np.nan, 4, 5],
            'B': [1, 2, 3, 4, 5]
        })
        result = mean_imputation(df)
        assert result.isnull().sum().sum() == 0
        print("✓ Imputation working")
        return True
    except Exception as e:
        print(f"✗ Imputation failed: {e}")
        return False


def test_univariate():
    """Test univariate analysis module."""
    print("Testing univariate analysis...")

    try:
        from eda_suite.univariate import compute_statistics

        data = np.random.normal(0, 1, 100)
        result = compute_statistics(data)
        assert hasattr(result, 'mean')
        assert hasattr(result, 'std')
        print("✓ Univariate analysis working")
        return True
    except Exception as e:
        print(f"✗ Univariate analysis failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 50)
    print("EDA Suite Validation Tests")
    print("=" * 50)
    print()

    tests = [
        test_imports,
        test_statistical_tests,
        test_hypothesis_testing,
        test_missing_data,
        test_imputation,
        test_univariate
    ]

    results = [test() for test in tests]

    print()
    print("=" * 50)
    print(f"Tests passed: {sum(results)}/{len(results)}")
    print("=" * 50)

    return all(results)


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
