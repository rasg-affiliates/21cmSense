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

    def test_multi_chunk(self):
        """Test that multiple chunks are handled correctly."""
        bls = np.array([[0.5, 0.5, 0], [1.5, 0.5, 0], [0.5, 1.5, 0], [1.5, 1.5, 0]]) * un.m
        wvlength = np.array([1]) * un.m
        freq = const.c / wvlength

        time_offsets = np.linspace(-2, 2, 25) * un.hour

        ugrid_edges = np.array([0, 1.0, 2.0])  # 2 cells along u

        req_mem = 8 * 4 * len(time_offsets) * 3 / 1024**3

        uvsum_chunk1 = ut.grid_baselines(
            coherent=True,
            baselines=bls,
            frequencies=freq,
            time_offsets=time_offsets,
            ugrid_edges=ugrid_edges,
            max_chunk_mem_gb=req_mem * 2,
        )

        uvsum_chunk2 = ut.grid_baselines(
            coherent=True,
            baselines=bls,
            frequencies=freq,
            time_offsets=time_offsets,
            ugrid_edges=ugrid_edges,
            max_chunk_mem_gb=req_mem / 4,
        )

        np.testing.assert_allclose(uvsum_chunk1, uvsum_chunk2)

    def test_bad_bls_shape(self):
        """Test that a bad baselines shape raises an error."""
        bls = np.array([[[0.5, 0.5], [1.5, 0.5], [0.5, 1.5], [1.5, 1.5]]]) * un.m
        wvlength = np.array([1]) * un.m
        freq = const.c / wvlength
        ugrid_edges = np.array([0, 1.0, 2.0])  # 2 cells along u
        with pytest.raises(ValueError, match="baselines must have shape"):
            ut.grid_baselines(
                coherent=True,
                baselines=bls,
                frequencies=freq,
                time_offsets=np.array([0]) * un.s,
                ugrid_edges=ugrid_edges,
            )

    def test_bad_vgrid_shape(self):
        """V-grid edges must be either full-plane or half-plane compatible."""
        bls = np.array([[0.5, 0.5, 0]]) * un.m
        freq = const.c / (np.array([1]) * un.m)

        with pytest.raises(ValueError, match="vgrid_edges must either be the same"):
            ut.grid_baselines(
                coherent=True,
                baselines=bls,
                frequencies=freq,
                time_offsets=np.array([0]) * un.s,
                ugrid_edges=np.array([0.0, 1.0, 2.0]),
                vgrid_edges=np.array([0.0, 1.0, 2.0, 3.0]),
            )

    def test_half_plane_requires_positive_northing(self):
        """Half-plane gridding expects ENU baselines with non-negative northing."""
        bls = np.array([[0.5, -0.5, 0]]) * un.m
        freq = const.c / (np.array([1]) * un.m)

        with pytest.raises(ValueError, match="positive Northing"):
            ut.grid_baselines(
                coherent=True,
                baselines=bls,
                frequencies=freq,
                time_offsets=np.array([0]) * un.s,
                ugrid_edges=np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
                vgrid_edges=np.array([0.0, 1.0, 2.0, 3.0]),
            )

    def test_grid_in_metres_requires_length_units(self):
        """If frequencies are omitted, the uv-grid must carry length units."""
        bls = np.array([[0.5, 0.5, 0]]) * un.m

        with pytest.raises(ValueError, match="ugrid_edges must have units of length"):
            ut.grid_baselines(
                coherent=True,
                baselines=bls,
                frequencies=None,
                time_offsets=np.array([0]) * un.s,
                ugrid_edges=np.array([0.0, 1.0, 2.0]),
            )


def test_convert_half_to_full_uv_plane_matches_full_gridding():
    """Half-plane conversion reproduces full-plane gridding for odd-sized grids."""
    from py21cmsense import Observatory
    from py21cmsense.baseline_filters import BaselineRange

    obs = Observatory.from_ska("LOW_FULL_AA4").clone(
        baseline_filters=BaselineRange(bl_max=250 * un.m)
    )
    time_offsets = np.arange(-3, 4, 1) * un.hour

    ugrid_edges = obs.xgrid_edges
    du = ugrid_edges[1] - ugrid_edges[0]
    # Observatory xgrid has an odd number of bins by construction.
    assert (len(ugrid_edges) - 1) % 2 == 1
    vgrid_edges = ugrid_edges[ugrid_edges + du >= 0]

    kw = {
        "coherent": True,
        "time_offsets": time_offsets,
        "ugrid_edges": ugrid_edges,
        "telescope_latitude": obs.latitude,
    }

    half_grid = ut.grid_baselines(
        baselines=obs.redundant_baseline_vectors,
        weights=obs.redundant_baseline_weights,
        vgrid_edges=vgrid_edges,
        **kw,
    )[0]

    full_grid = ut.grid_baselines(
        baselines=np.concatenate(
            [obs.redundant_baseline_vectors, -obs.redundant_baseline_vectors], axis=0
        ),
        weights=np.concatenate([obs.redundant_baseline_weights, obs.redundant_baseline_weights]),
        **kw,
    )[0]

    converted = ut.convert_half_to_full_uv_plane(half_grid)

    expected_half_sum = np.sum(obs.redundant_baseline_weights) * len(time_offsets)
    np.testing.assert_allclose(half_grid.sum(), expected_half_sum)

    np.testing.assert_allclose(converted, full_grid)
    np.testing.assert_allclose(converted.sum() / 2, half_grid.sum())


def test_convert_half_to_full_uv_plane_inverse_counts_handles_zeros():
    """Inverse-count mode remains finite when some half-plane cells are zero."""
    nu = 5
    nv = nu // 2 + 1
    rng = np.random.default_rng(1234 + 10 * nu)
    uv = rng.random((nu, nv, 2))
    uv[0, 0, :] = 0.0
    uv[-1, 0, :] = 0.0

    converted = ut.convert_half_to_full_uv_plane(uv, inverse_counts=True)

    assert converted.shape == (nu, nu, 2)
    assert np.all(np.isfinite(converted))


def test_convert_half_to_full_uv_plane_inverse_counts_matches_full_gridding():
    """Inverse-count conversion matches the full-plane inverse-count reference."""
    from py21cmsense import Observatory
    from py21cmsense.baseline_filters import BaselineRange

    obs = Observatory.from_ska("LOW_FULL_AA4").clone(
        baseline_filters=BaselineRange(bl_max=250 * un.m)
    )
    time_offsets = np.arange(-3, 4, 1) * un.hour

    ugrid_edges = obs.xgrid_edges
    du = ugrid_edges[1] - ugrid_edges[0]
    vgrid_edges = ugrid_edges[ugrid_edges + du >= 0]

    kw = {
        "coherent": True,
        "time_offsets": time_offsets,
        "ugrid_edges": ugrid_edges,
        "telescope_latitude": obs.latitude,
    }

    half_counts = ut.grid_baselines(
        baselines=obs.redundant_baseline_vectors,
        weights=obs.redundant_baseline_weights,
        vgrid_edges=vgrid_edges,
        **kw,
    )[0]

    full_counts = ut.grid_baselines(
        baselines=np.concatenate(
            [obs.redundant_baseline_vectors, -obs.redundant_baseline_vectors], axis=0
        ),
        weights=np.concatenate([obs.redundant_baseline_weights, obs.redundant_baseline_weights]),
        **kw,
    )[0]

    half_inv = np.zeros_like(half_counts, dtype=float)
    full_inv = np.zeros_like(full_counts, dtype=float)
    np.divide(1.0, half_counts, out=half_inv, where=half_counts > 0)
    np.divide(1.0, full_counts, out=full_inv, where=full_counts > 0)

    converted_inv = ut.convert_half_to_full_uv_plane(half_inv, inverse_counts=True)

    np.testing.assert_allclose(converted_inv, full_inv)


@pytest.mark.parametrize("nu", [4, 6])
def test_convert_half_to_full_uv_plane_even_nu_raises(nu: int):
    """Even-sized Nu inputs are rejected with guidance for the odd-grid workflow."""
    nv = nu // 2 + 1
    uv = np.ones((nu, nv))
    with pytest.raises(ValueError, match="requires an odd number of u-cells"):
        ut.convert_half_to_full_uv_plane(uv)


def test_convert_half_to_full_uv_plane_bad_shape():
    """Bad half-plane shape raises ValueError."""
    bad = np.ones((5, 4))
    with pytest.raises(ValueError, match="Input UV grid must have shape"):
        ut.convert_half_to_full_uv_plane(bad)
