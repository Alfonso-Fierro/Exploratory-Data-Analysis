"""
Input validation functions for EDA Suite.

Ensures data integrity following defensive programming principles.
"""

import numpy as np
import pandas as pd
from typing import Union


def validate_array(data: Union[np.ndarray, list]) -> np.ndarray:
    """
    Validate and convert input to numpy array.

    Args:
        data: Input data (array-like)

    Returns:
        Validated numpy array

    Raises:
        ValueError: If data is empty or invalid
    """
    arr = np.asarray(data)

    if arr.size == 0:
        raise ValueError("Input array cannot be empty")

    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError("Array must contain numeric values")

    return arr


def validate_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    """
    Validate pandas DataFrame.

    Args:
        data: Input DataFrame

    Returns:
        Validated DataFrame

    Raises:
        ValueError: If DataFrame is empty or invalid
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")

    if data.empty:
        raise ValueError("DataFrame cannot be empty")

    return data
