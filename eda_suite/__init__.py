"""
EDA Suite: Comprehensive Exploratory Data Analysis Library

A professional-grade library for automated statistical analysis, hypothesis testing,
missing data handling, imputation, discriminant analysis, factorial analysis,
and time series analysis.

Modules:
    statistical_tests: Normality, variance, and correlation tests
    hypothesis_testing: T-tests, ANOVA, chi-square, and more
    missing_data: Missing data pattern analysis
    imputation: Multiple imputation strategies
    discriminant: Linear Discriminant Analysis
    factorial: Factorial analysis methods
    timeseries: Time series analysis tools
    univariate: Single variable analysis
    bivariate: Two variable analysis
    multivariate: Multiple variable analysis
    visualization: Statistical visualization utilities
    utils: Helper functions and utilities
"""

__version__ = "1.0.0"
__author__ = "EDA Suite Contributors"

from eda_suite import (
    statistical_tests,
    hypothesis_testing,
    missing_data,
    imputation,
    discriminant,
    factorial,
    timeseries,
    univariate,
    bivariate,
    multivariate,
    visualization,
    utils
)

__all__ = [
    "statistical_tests",
    "hypothesis_testing",
    "missing_data",
    "imputation",
    "discriminant",
    "factorial",
    "timeseries",
    "univariate",
    "bivariate",
    "multivariate",
    "visualization",
    "utils"
]
