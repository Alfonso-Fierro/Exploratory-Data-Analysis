"""
Factor analysis and PCA implementation.

Dimension reduction and latent variable analysis.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import FactorAnalysis as SklearnFA
from sklearn.decomposition import PCA as SklearnPCA
from dataclasses import dataclass


@dataclass
class FactorResult:
    """Result of factor analysis."""

    loadings: np.ndarray
    variance_explained: np.ndarray
    scores: np.ndarray
    n_components: int


class FactorAnalysis:
    """Factor Analysis wrapper."""

    def __init__(self, n_components: int = 2):
        """
        Initialize Factor Analysis.

        Args:
            n_components: Number of factors
        """
        self.n_components = n_components
        self.model = SklearnFA(n_components=n_components)
        self.is_fitted = False

    def fit(self, X: np.ndarray) -> 'FactorAnalysis':
        """
        Fit factor analysis model.

        Args:
            X: Feature matrix

        Returns:
            Self for method chaining
        """
        self.model.fit(X)
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data to factor space.

        Args:
            X: Feature matrix

        Returns:
            Transformed data (factor scores)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        return self.model.transform(X)

    def get_results(self, X: np.ndarray) -> FactorResult:
        """
        Get factor analysis results.

        Args:
            X: Original feature matrix

        Returns:
            FactorResult with loadings and scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        scores = self.transform(X)
        variance = np.var(scores, axis=0)

        return FactorResult(
            loadings=self.model.components_.T,
            variance_explained=variance,
            scores=scores,
            n_components=self.n_components
        )


class PrincipalComponentAnalysis:
    """Principal Component Analysis wrapper."""

    def __init__(self, n_components: int = 2):
        """
        Initialize PCA.

        Args:
            n_components: Number of principal components
        """
        self.n_components = n_components
        self.model = SklearnPCA(n_components=n_components)
        self.is_fitted = False

    def fit(self, X: np.ndarray) -> 'PrincipalComponentAnalysis':
        """
        Fit PCA model.

        Args:
            X: Feature matrix

        Returns:
            Self for method chaining
        """
        self.model.fit(X)
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data to PC space.

        Args:
            X: Feature matrix

        Returns:
            Transformed data (PC scores)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        return self.model.transform(X)
