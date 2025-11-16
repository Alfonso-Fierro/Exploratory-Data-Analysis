"""
Linear and Quadratic Discriminant Analysis.

Classification methods based on Bayes' theorem.
"""

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis as SklearnLDA,
    QuadraticDiscriminantAnalysis as SklearnQDA
)
from dataclasses import dataclass
from typing import Tuple


@dataclass
class DiscriminantResult:
    """Result of discriminant analysis."""

    coefficients: np.ndarray
    means: np.ndarray
    explained_variance_ratio: np.ndarray
    predicted_classes: np.ndarray


class LinearDiscriminantAnalysis:
    """Linear Discriminant Analysis wrapper."""

    def __init__(self):
        """Initialize LDA model."""
        self.model = SklearnLDA()
        self.is_fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> 'LinearDiscriminantAnalysis':
        """
        Fit LDA model.

        Args:
            X: Feature matrix
            y: Target labels

        Returns:
            Self for method chaining
        """
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data to discriminant space.

        Args:
            X: Feature matrix

        Returns:
            Transformed data
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        return self.model.transform(X)

    def get_results(self) -> DiscriminantResult:
        """
        Get discriminant analysis results.

        Returns:
            DiscriminantResult with model information
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        return DiscriminantResult(
            coefficients=self.model.coef_,
            means=self.model.means_,
            explained_variance_ratio=self.model.explained_variance_ratio_,
            predicted_classes=self.model.classes_
        )


class QuadraticDiscriminantAnalysis:
    """Quadratic Discriminant Analysis wrapper."""

    def __init__(self):
        """Initialize QDA model."""
        self.model = SklearnQDA()
        self.is_fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> 'QuadraticDiscriminantAnalysis':
        """
        Fit QDA model.

        Args:
            X: Feature matrix
            y: Target labels

        Returns:
            Self for method chaining
        """
        self.model.fit(X, y)
        self.is_fitted = True
        return self
