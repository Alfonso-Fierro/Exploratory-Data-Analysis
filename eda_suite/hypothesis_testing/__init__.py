"""
Hypothesis testing module for EDA Suite.

Provides parametric and non-parametric hypothesis tests.
"""

from eda_suite.hypothesis_testing.parametric import (
    one_sample_ttest,
    two_sample_ttest,
    paired_ttest
)
from eda_suite.hypothesis_testing.anova import (
    one_way_anova,
    two_way_anova
)
from eda_suite.hypothesis_testing.nonparametric import (
    mann_whitney_test,
    wilcoxon_test,
    kruskal_wallis_test,
    friedman_test
)
from eda_suite.hypothesis_testing.categorical import (
    chi_square_test,
    fisher_exact_test,
    mcnemar_test
)

__all__ = [
    "one_sample_ttest",
    "two_sample_ttest",
    "paired_ttest",
    "one_way_anova",
    "two_way_anova",
    "mann_whitney_test",
    "wilcoxon_test",
    "kruskal_wallis_test",
    "friedman_test",
    "chi_square_test",
    "fisher_exact_test",
    "mcnemar_test"
]
