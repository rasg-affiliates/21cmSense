"""Test the _utils module."""

import numpy as np
import pytest
from astropy import constants as const
from astropy import units as un

from py21cmsense import _utils as ut


class TestGridBaselines:
    """Test the grid_baselines function."""

    @pytest.mark.parametrize("coherent", [True, False])
    @pytest.mark.parametrize("weight", [0, 1])
    @pytest.mark.parametrize("multiplicity", [1, 2])
    def test_trivial_one_bl_per_cell(self, coherent: bool, weight: float, multiplicity: int):
        """Test a trivial case where there is one baseline per uv cell.

        Also tests when multiplicity > 1, i.e. we just shove multiple copies of the same
        baselines in.
        """
        bls = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]] * multiplicity) * un.m
        wvlength = np.array([1]) * un.m
        freq = const.c / wvlength
        weights = np.ones(4 * multiplicity) * weight

        ugrid_edges = np.array([-0.5, 0.5, 1.5])  # 2 cells along u

        uvsum = ut.grid_baselines(
            coherent=coherent,
            baselines=bls,
            frequencies=freq,
            time_offsets=np.array([0]) * un.s,
            weights=weights,
            ugrid_edges=ugrid_edges,
        )

        np.testing.assert_allclose(
            uvsum, weight * (multiplicity if coherent else np.sqrt(multiplicity))
        )

    def test_multiple_time_offsets(self):
        """Test that multiple time offsets are handled correctly."""
        bls = np.array([[0.5, 0, 0], [0, 0.5, 0]]) * un.m
        wvlength = np.array([1]) * un.m
        freq = const.c / wvlength

        ugrid_edges = np.array([0, 1.0])  # 1 cell along u

        time_offsets = np.array([0, 1]) * un.s  # 1 second apart (no uv migration)

        uvsum = ut.grid_baselines(
            coherent=True,
            baselines=bls,
            frequencies=freq,
            time_offsets=time_offsets,
            ugrid_edges=ugrid_edges,
        )

        np.testing.assert_allclose(uvsum, 2)  # 2 times each.

    def test_multi_frequency_freqdep_grid(self):
        """Test that multi-frequency and frequency-dependent gridding works."""
        bls = np.array([[0.5, 0.5, 0], [1.5, 0.5, 0], [0.5, 1.5, 0], [1.5, 1.5, 0]]) * un.m
        wvlength = np.array([1, 2]) * un.m  # 2 frequencies
        freq = const.c / wvlength
        weights = np.ones(4)

        # Construct ugrid edges such that the baselines stay in the centre
        # of the bins at each wavelength.
        ugrid_edges = np.array(
            [[0, 1.0 * un.m / wv, 2.0 * un.m / wv] for wv in wvlength]
        )  # 1 cell along u
        print(ugrid_edges)
        uvsum = ut.grid_baselines(
            coherent=True,
            baselines=bls,
            frequencies=freq,
            time_offsets=np.array([0]) * un.s,
            weights=weights,
            ugrid_edges=ugrid_edges,
        )

        np.testing.assert_allclose(uvsum, 1)  # each baseline goes to a different cell.

    def test_multi_frequency_constant_grid(self):
        """Test that multi-frequency gridding with a constant grid works."""
        bls = np.array([[0.5, 0.5, 0], [1.5, 0.5, 0], [0.5, 1.5, 0], [1.5, 1.5, 0]]) * un.m
        wvlength = np.array([1, 2]) * un.m  # 2 frequencies
        freq = const.c / wvlength
        weights = np.ones(4)

        # Construct ugrid edges such that the baselines stay in the centre
        # of the bins at each wavelength.
        ugrid_edges = np.array([0, 1.0, 2.0])  # 2 cells along u

        uvsum = ut.grid_baselines(
            coherent=True,
            baselines=bls,
            frequencies=freq,
            time_offsets=np.array([0]) * un.s,
            weights=weights,
            ugrid_edges=ugrid_edges,
        )

        np.testing.assert_allclose(uvsum[0], 1)  # each baseline goes to its own cell
        np.testing.assert_allclose(
            uvsum[1], np.array([[4, 0], [0, 0]])
        )  # each baseline goes to a single cell
