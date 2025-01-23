"""Tests for the Observation class."""

import copy
import pickle

import numpy as np
import pytest
from astropy import units
from astropy.cosmology.units import littleh

from py21cmsense import GaussianBeam, Observation, Observatory


@pytest.fixture(scope="module")
def bm():
    return GaussianBeam(150.0 * units.MHz, dish_size=14 * units.m)


@pytest.fixture(scope="module", params=["earth", "moon"])
def wd(request):
    return request.param


@pytest.fixture(scope="module")
def observatory(bm, wd):
    return Observatory(
        antpos=np.array([[0, 0, 0], [14, 0, 0], [28, 0, 0], [70, 0, 0], [0, 14, 0], [23, -45, 0]])
        * units.m,
        latitude=-32 * units.deg,
        beam=bm,
        world=wd,
    )


def test_units(observatory):
    obs = Observation(observatory=observatory)

    assert obs.time_per_day.unit == units.hour
    assert obs.lst_bin_size.to("min").unit == units.min
    assert obs.integration_time.to("s").unit == units.s
    assert obs.bandwidth.to("MHz").unit == units.MHz
    assert obs.bl_min.to("m").unit == units.m
    assert obs.bl_max.to("m").unit == units.m
    assert obs.tsky_amplitude.to("mK").unit == units.mK
    assert obs.tsky_ref_freq.to("MHz").unit == units.MHz

    assert obs.frequency == observatory.frequency
    assert obs.n_lst_bins > 1
    assert obs.Tsky.to("mK").unit == units.mK
    assert obs.Tsys.to("mK").unit == units.mK
    assert obs.Trms.to("mK").unit == units.mK
    assert 6 < obs.redshift < 12
    assert obs.kparallel.unit == littleh / units.Mpc
    assert obs.total_integration_time.to("s").unit == units.s
    assert len(obs.ugrid_edges) == len(obs.ugrid) + 1
    assert obs.clone() == obs


def test_pickle(observatory):
    obs = Observation(observatory=observatory)

    string_rep = pickle.dumps(obs)
    obs2 = pickle.loads(string_rep)
    assert obs == obs2


def test_uvcov(observatory):
    coherent_obs = Observation(observatory=observatory, coherent=True)

    incoherent_obs = Observation(observatory=observatory, coherent=False)

    assert np.all(coherent_obs.uv_coverage >= incoherent_obs.uv_coverage)


def test_equality(observatory):
    new_observatory = copy.deepcopy(observatory)
    assert new_observatory == observatory


def test_from_yaml(observatory):
    rng = np.random.default_rng(1234)
    obs = Observation.from_yaml(
        {
            "observatory": {
                "antpos": rng.random((20, 3)) * units.m,
                "beam": {
                    "class": "GaussianBeam",
                    "frequency": 150 * units.MHz,
                    "dish_size": 14 * units.m,
                },
            }
        }
    )
    assert obs.observatory.antpos.shape == (20, 3)

    with pytest.raises(ValueError, match="yaml_file must be a string filepath"):
        Observation.from_yaml(3)


def test_huge_time_per_day_size(observatory: Observatory, wd):
    tpd = 25 * units.hour if wd == "earth" else 682.5 * units.hour
    with pytest.raises(ValueError, match="time_per_day should be between 0 and"):
        Observation(observatory=observatory, time_per_day=tpd)


def test_huge_track_size(observatory: Observatory, wd):
    tck = 25 * units.hour if wd == "earth" else 682.5 * units.hour
    with pytest.raises(ValueError, match="track should be between 0 and"):
        Observation(observatory=observatory, track=tck)


def test_huge_lst_bin_size(observatory: Observatory, wd):
    lst = 23 * units.hour if wd == "earth" else 627.9 * units.hour
    with pytest.raises(ValueError, match="lst_bin_size must be <= time_per_day"):
        Observation(observatory=observatory, lst_bin_size=lst)

    lst2 = 25 * units.hour if wd == "earth" else 682.5 * units.hour
    with pytest.raises(ValueError, match="lst_bin_size should be between 0 and"):
        Observation(observatory=observatory, lst_bin_size=lst2)


def test_huge_integration_time(observatory: Observatory):
    with pytest.raises(ValueError, match="integration_time must be <= lst_bin_size"):
        Observation(
            observatory=observatory,
            lst_bin_size=1 * units.hour,
            integration_time=2 * units.hour,
        )


def test_trcv_func(observatory: Observatory):
    observatory = observatory.clone(Trcv=lambda f: (f / units.MHz) * 10 * units.mK)

    obs = Observation(
        observatory=observatory,
    )
    assert obs.Trcv.unit == units.K
    assert obs.Trcv == (observatory.beam.frequency / units.MHz) * 0.01 * units.K


def test_non_zenith_pointing(bm):
    """Test that not pointing at zenith gives different results."""
    observatory = Observatory(
        antpos=np.array([[0, 0, 0], [14, 0, 0], [28, 0, 0], [70, 0, 0], [0, 14, 0], [23, -45, 0]])
        * units.m,
        latitude=-32 * units.deg,
        beam=bm,
    )
    at_zenith = Observation(observatory=observatory)

    almost_zenith = Observation(
        observatory=observatory, phase_center_dec=observatory.latitude * 0.99
    )
    np.testing.assert_allclose(at_zenith.uv_coverage, almost_zenith.uv_coverage)

    not_zenith = Observation(
        observatory=observatory,
        phase_center_dec=observatory.latitude + 45 * units.deg,
    )
    assert not np.allclose(not_zenith.uv_coverage, at_zenith.uv_coverage)


def test_non_zenith_pointing_only_ew(bm):
    """Test that not pointing at zenith gives the SAME results if all baselines are EW."""
    observatory = Observatory(
        antpos=np.array([[0, 0, 0], [14, 0, 0], [28, 0, 0], [70, 0, 0]]) * units.m,
        latitude=-32 * units.deg,
        beam=bm,
    )
    at_zenith = Observation(observatory=observatory)

    not_zenith = Observation(
        observatory=observatory,
        phase_center_dec=observatory.latitude + 45 * units.deg,
    )
    np.testing.assert_allclose(not_zenith.uv_coverage, at_zenith.uv_coverage)
