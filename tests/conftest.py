"""Pytest configuration and fixtures for the blood-culture-outcome-classification tests."""

import sys
from pathlib import Path

import pytest
import numpy as np
import pandas as pd

# Add the python_scripts directory to the path so tests can import utilities
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "python_scripts"))


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "feature1": np.random.randn(n),
        "feature2": np.random.randn(n),
        "feature3": np.random.randn(n),
        "BLOOD_CULTURE_POSITIVE": np.random.randint(0, 2, n),
    })


@pytest.fixture
def sample_features():
    """Sample feature list."""
    return ["feature1", "feature2", "feature3"]


@pytest.fixture
def sample_binary_labels():
    """Sample binary classification labels."""
    np.random.seed(42)
    return np.random.randint(0, 2, 100)


@pytest.fixture
def sample_probabilities():
    """Sample probability predictions."""
    np.random.seed(42)
    return np.random.rand(100)
