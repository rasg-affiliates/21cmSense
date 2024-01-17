import pytest

import copy
import numpy as np
import pickle
from astropy import units
from astropy.cosmology.units import littleh

from py21cmsense import GaussianBeam, Observation, Observatory


@pytest.fixture(scope="module")
def bm():
    return GaussianBeam(150.0 * units.MHz, dish_size=14 * units.m)


@pytest.fixture(scope="module")
def observatory(bm):
    return Observatory(
        antpos=np.array([[0, 0, 0], [14, 0, 0], [28, 0, 0], [70, 0, 0]]) * units.m,
        beam=bm,
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
    obs = Observation.from_yaml(
        {
            "observatory": {
                "antpos": np.random.random((20, 3)) * units.m,
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


def test_huge_lst_bin_size(observatory: Observatory):
    with pytest.raises(ValueError, match="lst_bin_size must be <= time_per_day"):
        Observation(observatory=observatory, lst_bin_size=23 * units.hour)


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
