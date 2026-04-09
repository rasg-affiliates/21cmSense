"""Test the observatory module."""

import re
from pathlib import Path

import numpy as np
import pytest
import pyuvdata
from astropy import units
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time

from py21cmsense import Observatory
from py21cmsense.baseline_filters import BaselineRange
from py21cmsense.beam import GaussianBeam
from py21cmsense.data import PATH


@pytest.fixture(scope="module")
def bm():
    return GaussianBeam(dish_size=14 * units.m)


def test_antpos(bm):
    a = Observatory(antpos=np.zeros((10, 3)) * units.m, beam=bm)
    assert a.antpos.unit == units.m

    assert np.all(a.baselines_metres == 0)

    # If bad units given, should raise error.
    with pytest.raises(units.UnitConversionError):
        Observatory(antpos=np.zeros((10, 3)) * units.s, beam=bm)

    # Need more than one antenna
    with pytest.raises(ValueError, match="antpos must have at least two antennas"):
        Observatory(antpos=np.zeros((1, 3)) * units.m, beam=bm)

    with pytest.raises(ValueError, match="antpos must be a 2D array"):
        Observatory(antpos=np.zeros(10) * units.m, beam=bm)

    with pytest.raises(ValueError, match=re.escape("antpos must have shape (Nants, 3)")):
        Observatory(antpos=np.zeros((10, 2)) * units.m, beam=bm)


def test_observatory_class(bm):
    a = Observatory(antpos=np.zeros((3, 3)) * units.m, beam=bm)
    b = a.clone()
    assert a == b


def test_Trcv(bm):
    a = Observatory(antpos=np.zeros((3, 3)) * units.m, beam=bm, Trcv=10 * units.mK)
    assert a.Trcv.unit == units.mK


def test_Trcv_func(bm):
    a = Observatory(
        antpos=np.zeros((3, 3)) * units.m,
        beam=bm,
        Trcv=lambda f: (f / units.MHz) * 10 * units.mK,
    )
    assert a.Trcv(7 * units.Hz).unit.is_equivalent(units.K)


def test_Trcv_func_bad(bm):
    with pytest.raises(ValueError, match="Trcv function must return a temperature"):
        Observatory(antpos=np.zeros((3, 3)) * units.m, beam=bm, Trcv=lambda f: 3)


def test_Trcv_func_bad_signature(bm):
    with pytest.raises(ValueError, match="Trcv function must take a frequency Quantity"):
        Observatory(
            antpos=np.zeros((3, 3)) * units.m,
            beam=bm,
            Trcv=lambda f: 1 / 0,
        )


def test_observatory(bm):
    a = Observatory(antpos=np.zeros((3, 3)) * units.m, beam=bm)
    assert a.baselines_metres.shape == (3, 3)
    assert a.baseline_lengths.shape == (3,)
    assert np.all(a.baseline_lengths == 0)

    b = Observatory(antpos=np.array([[0, 0, 0], [1, 0, 0], [3, 0, 0]]) * units.m, beam=bm)
    assert len(b.redundant_baseline_groups) == 3
    with pytest.raises(AssertionError):
        b.time_offsets_from_obs_int_time(
            integration_time=3 * units.hour * 1.1, observation_duration=3 * units.hour
        )

    assert (
        len(
            b.time_offsets_from_obs_int_time(
                integration_time=3 * units.hour / 1.05, observation_duration=3 * units.hour
            )
        )
        == 2
    )


def test_min_max_antpos(bm):
    a = Observatory(
        antpos=np.array([np.linspace(0, 50, 11), np.zeros(11), np.zeros(11)]).T * units.m,
        beam=bm,
        min_antpos=7 * units.m,
    )

    assert len(a.antpos) == 9

    a = Observatory(
        antpos=np.array([np.linspace(0, 50, 11), np.zeros(11), np.zeros(11)]).T * units.m,
        beam=bm,
        max_antpos=10 * units.m,
    )

    assert len(a.antpos) == 2


def test_from_uvdata(bm):
    uv = pyuvdata.UVData()
    uv.telescope.antenna_positions = (
        np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [40, 0, 40]]) * units.m
    )
    uv.telescope.location = [x.value for x in EarthLocation.from_geodetic(0, 0).to_geocentric()]
    uv.telescope.antenna_positions = (
        np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [40, 0, 40]]) * units.m
    )
    uv.telescope.location = EarthLocation.from_geodetic(0, 0)

    a = Observatory.from_uvdata(uvdata=uv, beam=bm)
    assert np.all(a.antpos == uv.telescope.antenna_positions)


def test_different_antpos_loaders(tmp_path: Path):
    antpos = np.array([[0, 0, 0], [14, 0, 0], [28, 0, 0], [70, 0, 0]])

    np.save(tmp_path / "antpos.npy", antpos)
    np.savetxt(tmp_path / "antpos.txt", antpos)

    beamtxt = """
    beam:
        class: GaussianBeam
        dish_size: !astropy.units.Quantity
            unit: !astropy.units.Unit {unit: m}
            value: 14.0
    """

    yamlnpy = f"""
    antpos: !astropy.units.Quantity
        unit: !astropy.units.Unit {{unit: m}}
        value: !npy {tmp_path}/antpos.npy
    {beamtxt}
    """

    with open(tmp_path / "npy.yml", "w") as fl:
        fl.write(yamlnpy)

    obsnpy = Observatory.from_yaml(tmp_path / "npy.yml")

    yamltxt = f"""
    antpos: !astropy.units.Quantity
        unit: !astropy.units.Unit {{unit: m}}
        value: !txt {tmp_path}/antpos.txt
    {beamtxt}
    """
    with open(tmp_path / "txt.yml", "w") as fl:
        fl.write(yamltxt)

    obstxt = Observatory.from_yaml(tmp_path / "txt.yml")

    assert obsnpy == obstxt


def test_from_yaml(bm):
    rng = np.random.default_rng(1234)
    obs = Observatory.from_yaml(
        {
            "antpos": rng.random((20, 3)) * units.m,
            "beam": {
                "class": "GaussianBeam",
                "dish_size": 14 * units.m,
            },
        }
    )
    assert obs.beam == bm

    with pytest.raises(ValueError, match="yaml_file must be a string filepath"):
        Observatory.from_yaml(3)


def test_from_yaml_with_max_antpos_filtering(bm):
    obs = Observatory.from_yaml(
        {
            "antpos": np.array([[0, 0, 0], [5, 0, 0], [50, 0, 0]]) * units.m,
            "max_antpos": 10 * units.m,
            "beam": {
                "class": "GaussianBeam",
                "dish_size": 14 * units.m,
            },
        }
    )
    assert obs.beam == bm
    assert len(obs.antpos) == 2


def test_from_ska():
    pytest.importorskip("ska_ost_array_config")

    from ska_ost_array_config import UVW, get_subarray_template
    from ska_ost_array_config.simulation_utils import simulate_observation

    obs = Observatory.from_ska("LOW_FULL_AASTAR")
    low_aastar = get_subarray_template("LOW_FULL_AASTAR")
    assert obs.antpos.shape == low_aastar.array_config.xyz.data.shape
    Observatory.from_ska(subarray_template="MID_FULL_AASTAR")
    obs = Observatory.from_ska(subarray_template="LOW_FULL_AA4")
    low_aa4 = get_subarray_template("LOW_FULL_AA4")
    assert obs.antpos.shape == low_aa4.array_config.xyz.data.shape
    obs = Observatory.from_ska(
        subarray_template="LOW_INNER_R350M_AASTAR",
        Trcv=100.0 * units.K,
        exclude_stations="C1,C2",
    )
    low_custom = get_subarray_template("LOW_INNER_R350M_AASTAR", exclude_stations="C1,C2")
    assert obs.antpos.shape == low_custom.array_config.xyz.data.shape

    # Simulate visibilities and retreive the UVW values
    ref_time = Time.now()
    zenith = SkyCoord(
        alt=90 * units.deg,
        az=0 * units.deg,
        frame="altaz",
        obstime=ref_time,
        location=low_custom.array_config.location,
    ).icrs
    vis = simulate_observation(
        array_config=low_custom.array_config,
        phase_centre=zenith,
        start_time=ref_time,
        ref_freq=50e6,  # Dummy value. We are after uvw values in [m]
        chan_width=1e3,  # Dummy value. We are after uvw values in [m]
        n_chan=1,
    )
    uvw = UVW.UVW(vis, ignore_autocorr=False)
    uvw_m = uvw.uvdist_m
    assert np.allclose(obs.bl_max, uvw_m.max() * units.m)


def test_get_redundant_baselines(bm):
    a = Observatory(antpos=np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]]) * units.m, beam=bm)

    reds = a.redundant_baseline_groups
    assert len(reds) == 2  # len-1, len-2

    a = a.clone(baseline_filters=BaselineRange(bl_max=1.5 * units.m))
    print(a.baselines_metres)
    reds = a.redundant_baseline_groups
    assert len(reds) == 1  # len-1


def test_no_up_coordinate(tmp_path: Path):
    mwafl = PATH / "antpos" / "mwa_phase2_compact_antpos.txt"
    enu = np.genfromtxt(mwafl)

    # Save with only EN coordinates
    with open(tmp_path / "mwa_antpos.txt", "w") as fl:
        np.savetxt(fl, enu[:, :2])

    new_yaml = f"""
antpos: !astropy.units.Quantity
  value: !txt "{tmp_path}/mwa_antpos.txt"
  unit: !astropy.units.Unit {{unit: m}}
beam:
  class: GaussianBeam
  dish_size: !astropy.units.Quantity
    unit: !astropy.units.Unit {{unit: m}}
    value: 35
latitude: !astropy.units.Quantity
  unit: !astropy.units.Unit {{unit: rad}}
  value: -0.4681819
Trcv: !astropy.units.Quantity
  unit: !astropy.units.Unit {{unit: K}}
  value: 100
"""

    with open(tmp_path / "mwa.yaml", "w") as fl:
        fl.write(new_yaml)

    obs = Observatory.from_yaml(tmp_path / "mwa.yaml")
    assert np.all(obs.antpos[:, 2] == 0)


def test_projected_baselines_metres(bm):
    rng = np.random.default_rng(1234)
    obs = Observatory(antpos=rng.normal(loc=0, scale=50, size=(20, 3)) * units.m, beam=bm)
    bl_coords = obs.redundant_baseline_vectors

    time_offsets = obs.time_offsets_from_obs_int_time(
        integration_time=3 / 10 * units.hour, observation_duration=3 * units.hour
    )

    proj_bls = obs.projected_baselines(
        baselines=bl_coords,
        time_offset=time_offsets,
    )
    assert proj_bls.unit == units.m
    assert proj_bls.shape == (len(bl_coords), len(time_offsets), 3)


def test_grid_baselines_frequency_defaults_observation_duration(bm):
    obs = Observatory(antpos=np.array([[0, 0, 0], [14, 0, 0], [28, 0, 0]]) * units.m, beam=bm)

    grid = obs.grid_baselines(
        coherent=True,
        frequency=150 * units.MHz,
        integration_time=10 * units.s,
    )

    assert grid.ndim == 2
