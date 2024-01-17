"""Test baseline_filters module."""

import pytest

import numpy as np
from astropy import units as un

from py21cmsense.baseline_filters import BaselineRange

# Test IDs for parametrization
happy_path_ids = [
    "east-west_min",
    "north-south_min",
    "magnitude_min",
    "east-west_max",
    "north-south_max",
    "magnitude_max",
]

edge_case_ids = [
    "east-west_zero",
    "north-south_zero",
    "magnitude_zero",
]

error_case_ids = [
    "bl_max_less_than_bl_min",
    "invalid_direction",
    "non_length_input",
]

# Happy path test values
happy_path_values = [
    (0 * un.m, np.inf * un.m, "ew", np.array([1, 0, 0]) * un.m, True),
    (0 * un.m, np.inf * un.m, "ns", np.array([0, 1, 0]) * un.m, True),
    (0 * un.m, np.inf * un.m, "mag", np.array([1, 1, 0]) * un.m, True),
    (1 * un.m, np.inf * un.m, "ew", np.array([2, 0, 0]) * un.m, True),
    (1 * un.m, np.inf * un.m, "ns", np.array([0, 2, 0]) * un.m, True),
    (1 * un.m, np.inf * un.m, "mag", np.array([1, 1, 0]) * un.m, True),
]

# Edge case test values
edge_case_values = [
    (0 * un.m, np.inf * un.m, "ew", np.array([0, 0, 0]) * un.m, True),
    (0 * un.m, np.inf * un.m, "ns", np.array([0, 0, 0]) * un.m, True),
    (0 * un.m, np.inf * un.m, "mag", np.array([0, 0, 0]) * un.m, True),
]

# Error case test values
error_case_values = [
    (2 * un.m, 1 * un.m, "mag", np.array([1, 1, 0]) * un.m),
    (0 * un.m, np.inf * un.m, "invalid", np.array([1, 1, 0]) * un.m),
    (0 * un.m, np.inf * un.m, "ew", np.array([1, 1, 0])),
]


@pytest.mark.parametrize(
    "bl_min, bl_max, direction, baseline, expected",
    happy_path_values,
    ids=happy_path_ids,
)
def test_happy_path(bl_min, bl_max, direction, baseline, expected):
    # Arrange
    baseline_range = BaselineRange(bl_min=bl_min, bl_max=bl_max, direction=direction)

    # Act
    result = baseline_range(baseline)

    # Assert
    assert result == expected


@pytest.mark.parametrize(
    "bl_min, bl_max, direction, baseline, expected", edge_case_values, ids=edge_case_ids
)
def test_edge_cases(bl_min, bl_max, direction, baseline, expected):
    # Arrange
    baseline_range = BaselineRange(bl_min=bl_min, bl_max=bl_max, direction=direction)

    # Act
    result = baseline_range(baseline)

    # Assert
    assert result == expected


@pytest.mark.parametrize(
    "bl_min, bl_max, direction, baseline", error_case_values, ids=error_case_ids
)
def test_error_cases(bl_min, bl_max, direction, baseline):
    # Arrange
    with pytest.raises(ValueError):
        baseline_range = BaselineRange(
            bl_min=bl_min, bl_max=bl_max, direction=direction
        )
        baseline_range(baseline)
