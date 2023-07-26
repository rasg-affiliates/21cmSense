"""Utility functions for 21cmSense."""
import attr
import h5py
import importlib
import inspect
import numpy as np
import tqdm
import yaml
from astropy import units as un
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from pyuvdata import utils as uvutils

from . import config


def between(xmin, xmax):
    """Return an attrs validation function that checks a number is within bounds."""

    def validator(instance, att, val):
        assert xmin <= val <= xmax

    return validator


def positive(instance, att, x):
    """An attrs validator that checks a value is positive."""
    assert x > 0, "must be positive"


def nonnegative(instance, att, x):
    """An attrs validator that checks a value is non-negative."""
    assert x >= 0, "must be non-negative"


def find_nearest(array, value):
    """Find closest value in `array` to `value`."""
    return np.abs(array.reshape(-1, 1) - value).argmin(0)


@un.quantity_input
def phase_past_zenith(time_past_zenith: un.day, bls_enu: np.ndarray, latitude):
    """Compute UVWs phased to a point rotated from zenith by a certain amount of time.

    This function specifies a longitude and time of observation without loss of
    generality -- all that matters is the time since a hypothetical point was at zenith,
    and the latitude of the array.

    Parameters
    ----------
    time_past_zenith
        The time passed since the point was at zenith. If float, assumed to be in units
        of days.
    uvws0 : array
        The UVWs when phased to zenith.
    latitude
        The latitude of the center of the array, in radians.

    Returns
    -------
    uvws
        The array of UVWs correctly phased.
    """
    # Generate ra/dec of zenith at time in the phase_frame coordinate system
    # to use for phasing
    telescope_location = EarthLocation.from_geodetic(lon=0, lat=latitude)

    # JD is arbitrary
    jd = 2454600

    zenith_coord = SkyCoord(
        alt=90 * un.deg,
        az=0 * un.deg,
        obstime=Time(jd, format="jd"),
        frame="altaz",
        location=telescope_location,
    )
    zenith_coord = zenith_coord.transform_to("icrs")

    phase_coords = SkyCoord(
        ra=zenith_coord.ra,
        dec=zenith_coord.dec,
        obstime=zenith_coord.obstime + time_past_zenith,
        frame="icrs",
        location=telescope_location,
    )
    lsts = phase_coords.obstime.sidereal_time("apparent", longitude=0.0).rad

    if not hasattr(lsts, "__len__"):
        lsts = np.array([lsts])

    # Now make everything nbls * ntimes big.
    app_ra = zenith_coord.ra.rad * np.ones(len(bls_enu) * len(lsts))
    app_dec = zenith_coord.dec.rad * np.ones(len(bls_enu) * len(lsts))
    _lsts = np.tile(lsts, len(bls_enu))
    uvws = np.repeat(bls_enu, len(lsts), axis=0)

    out = uvutils.calc_uvw(
        app_ra=app_ra,
        app_dec=app_dec,
        lst_array=_lsts,
        uvw_array=uvws,
        telescope_lat=latitude.to_value("rad"),
        telescope_lon=0.0,
        from_enu=True,
        use_ant_pos=False,
    )
    return out.reshape((len(bls_enu), len(lsts), 3))
