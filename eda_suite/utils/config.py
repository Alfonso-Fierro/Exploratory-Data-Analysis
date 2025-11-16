"""
Configuration dataclasses for EDA Suite modules.

Following SOLID principles by encapsulating configuration logic.
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class TestConfig:
    """Configuration for statistical tests."""

    alpha: float = 0.05
    alternative: str = "two-sided"

    def __post_init__(self):
        """Validate configuration parameters."""
        if not 0 < self.alpha < 1:
            raise ValueError("Alpha must be between 0 and 1")

        valid_alternatives = ["two-sided", "less", "greater"]
        if self.alternative not in valid_alternatives:
            raise ValueError(f"Alternative must be one of {valid_alternatives}")


@dataclass
class ImputationConfig:
    """Configuration for data imputation."""

    strategy: str = "mean"
    n_neighbors: int = 5
    max_iter: int = 10

    def __post_init__(self):
        """Validate configuration parameters."""
        valid_strategies = ["mean", "median", "mode", "knn", "iterative", "mice"]
        if self.strategy not in valid_strategies:
            raise ValueError(f"Strategy must be one of {valid_strategies}")

        if self.n_neighbors < 1:
            raise ValueError("n_neighbors must be positive")


@dataclass
class VisualizationConfig:
    """Configuration for visualization settings."""

    figsize: tuple = (10, 6)
    style: str = "whitegrid"
    palette: str = "deep"
    context: str = "notebook"

    def __post_init__(self):
        """Validate configuration parameters."""
        if len(self.figsize) != 2:
            raise ValueError("figsize must be a tuple of (width, height)")

        valid_contexts = ["paper", "notebook", "talk", "poster"]
        if self.context not in valid_contexts:
            raise ValueError(f"Context must be one of {valid_contexts}")
