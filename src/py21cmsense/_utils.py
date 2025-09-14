"""Utility functions for 21cmSense."""

import numpy as np
import tqdm
from astropy import units as un
from astropy.constants import c as speed_of_light
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from fast_histogram import histogram2d
from lunarsky import MoonLocation
from lunarsky import SkyCoord as LunarSkyCoord
from lunarsky import Time as LTime
from pyuvdata import utils as uvutils

from . import config
from . import units as tp


def between(xmin, xmax):
    """Return an attrs validation function that checks a number is within bounds."""

    def validator(instance, att, val):
        assert xmin <= val <= xmax

    return validator


def positive(instance, att, x):
    """Check that a value is positive.

    This is an attrs validator.
    """
    assert x > 0, "must be positive"


def nonnegative(instance, att, x):
    """Check that a value is non-negative.

    This is an attrs validator.
    """
    assert x >= 0, "must be non-negative"


def find_nearest(array, value):
    """Find closest value in `array` to `value`."""
    return np.abs(array.reshape(-1, 1) - value).argmin(0)


@un.quantity_input
def phase_past_zenith(
    time_past_zenith: un.hour,
    bls_enu: np.ndarray,
    latitude: float,
    world: str = "earth",
    phase_center_dec: un.rad = None,
    use_apparent: bool = True,
):
    """Compute UVWs phased to a point rotated from zenith by a certain amount of time.

    This function specifies a longitude and time of observation without loss of
    generality -- all that matters is the time since a hypothetical point was at zenith,
    and the latitude of the array.

    Parameters
    ----------
    time_past_zenith
        The time passed since the point was at its closest to zenith. Must be a
        quantity with time units.
    bls_enu
        An (Nbls, 3)-shaped array containing the ENU coordinates of the baseline
        vectors (equivalent to the UVWs if phased to zenith).
    latitude
        The latitude of the center of the array, in radians.
    world
        Whether the telescope is on the Earth or Moon.
    phase_center_dec
        If given, the declination of the phase center. If not given, it is set to the
        latitude of the array (i.e. the phase center passes through zenith).
    use_apparent
        Whether to use the apparent coordinates of the phase center (i.e. after
        accounting for nutation and precession etc.)

    Returns
    -------
    uvws
        The array of UVWs correctly phased.
    """
    # Generate ra/dec of zenith at time in the phase_frame coordinate system
    # to use for phasing
    if world == "earth":
        telescope_location = EarthLocation.from_geodetic(lon=0, lat=latitude)
    else:
        telescope_location = MoonLocation.from_selenodetic(lon=0, lat=latitude)

    # JD is arbitrary
    jd = 2454600

    if world == "earth":
        tm = Time(jd, format="jd", location=telescope_location)

        phase_center_coord = SkyCoord(
            alt=90 * un.deg,
            az=0 * un.deg,
            obstime=tm,
            frame="altaz",
            location=telescope_location,
        )
    else:
        tm = LTime(jd, format="jd", location=telescope_location)

        phase_center_coord = LunarSkyCoord(
            alt=90 * un.deg,
            az=0 * un.deg,
            obstime=tm,
            frame="lunartopo",
            location=telescope_location,
        )

    phase_center_coord = phase_center_coord.transform_to("icrs")

    if phase_center_dec is not None:
        phase_center_coord = phase_center_coord.__class__(
            ra=phase_center_coord.ra,
            dec=phase_center_dec,
            obstime=tm,
            frame=phase_center_coord.frame,
            location=telescope_location,
        )

    obstimes = phase_center_coord.obstime + time_past_zenith
    lsts = obstimes.sidereal_time("apparent", longitude=0.0).rad

    if not hasattr(lsts, "__len__"):
        lsts = np.array([lsts])

    if use_apparent:
        app_ra, app_dec = uvutils.phasing.calc_app_coords(
            lon_coord=phase_center_coord.ra.to_value("rad"),
            lat_coord=phase_center_coord.dec.to_value("rad"),
            time_array=obstimes.utc.jd,
            telescope_loc=telescope_location,
        )

        app_ra = np.tile(app_ra, len(bls_enu))
        app_dec = np.tile(app_dec, len(bls_enu))

    else:
        app_ra = phase_center_coord.ra.to_value("rad") * np.ones(len(bls_enu) * len(lsts))
        app_dec = phase_center_coord.dec.to_value("rad") * np.ones(len(bls_enu) * len(lsts))

    # Now make everything nbls * ntimes big.
    _lsts = np.tile(lsts, len(bls_enu))
    uvws = np.repeat(bls_enu, len(lsts), axis=0)

    out = uvutils.phasing.calc_uvw(
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


def project_baselines(
    baselines: tp.Length,
    telescope_latitude: tp.Angle,
    time_offsets: tp.Time = 0 * un.hour,
    phase_center_dec: tp.Angle | None = None,
    world: str = "earth",
) -> np.ndarray:
    """Compute *projected* baseline vectors in metres.

    Phased to a point that has rotated off zenith by some time_offset.

    Parameters
    ----------
    baselines
        The baseline co-ordinates to project, assumed to be in metres.
        If not provided, uses all baselines of the observatory.
        Shape of the array can be (N,N,3) or (N, 3).
        The co-ordinates are expected to be in ENU.
    time_offset
        The amount of time elapsed since the phase center was at zenith.
        Assumed to be in days unless otherwise defined. May be negative.
    phase_center_dec
        The declination of the phase center of the observation. By default, the
        same as the latitude of the array.

    Returns
    -------
    An array the same shape as :attr:`baselines_metres`, but phased to the
    new phase centre.
    """
    orig_shape = baselines.shape

    bls = baselines.reshape((-1, 3))

    out = phase_past_zenith(
        time_past_zenith=time_offsets,
        bls_enu=bls,
        latitude=telescope_latitude,
        world=world,
        phase_center_dec=phase_center_dec,
    )

    out = out.reshape(*orig_shape[:-1], np.size(time_offsets), orig_shape[-1])
    if np.size(time_offsets) == 1:
        out = out.squeeze(-2)

    return out


def grid_baselines(
    coherent: bool,
    baselines: tp.Length,
    weights: np.ndarray,
    time_offsets: tp.Time,
    frequencies: tp.Frequency,
    ugrid_edges: np.ndarray,
    phase_center_dec: tp.Angle | None = None,
    telescope_latitude: tp.Angle = 0 * un.deg,
    world: str = "earth",
) -> np.ndarray:
    """
    Grid baselines onto a pre-determined uvgrid, accounting for earth rotation.

    Parameters
    ----------
    baselines : array_like, optional
        The baseline co-ordinates to project, assumed to be in metres.
        If not provided, calculates effective baselines by finding redundancies on
        all baselines in the observatory. Shape of the array can be (N,N,3) or (N, 3).
        The co-ordinates are expected to be in ENU. If `baselines` is provided,
        `weights` must also be provided.
    weights: array_like, optional
        An array of the same length as `baselines`, giving the number of independent
        baselines at each co-ordinate. If not provided, calculates effective
        baselines by finding redundancies on all baselines in the observatory.
        If `baselines` is provided, `weights` must also be provided.
    integration_time : float or Quantity, optional
        The amount of time integrated into a snapshot visibility, assumed
        to be in seconds.
    baseline_filters
        A function that takes a single value: a length-3 array of baseline co-ordinates,
        and returns a bool indicating whether to include the baseline. Built-in filters
        are provided in the :mod:`~baseline_filters` module.
    observation_duration : float or Quantity, optional
        Amount of time in a single (coherent) LST bin, assumed to be in minutes.
    ndecimals : int, optional
        Number of decimals to which baselines must match to be considered redundant.

    Returns
    -------
    array :
        Shape [n_baseline_groups, Nuv, Nuv]. The coherent sum of baselines within
        grid cells given by :attr:`ugrid`. One can treat different baseline groups
        independently, or sum over them.

    See Also
    --------
    grid_baselines_coherent :
        Coherent sum over baseline groups of the output of this method.
    grid_baseline_incoherent :
        Incoherent sum over baseline groups of the output of this method.
    """
    if weights is None:
        weights = np.ones(len(baselines))

    if coherent:
        weights = np.repeat(weights, len(time_offsets))

    proj_bls = project_baselines(
        baselines,
        time_offsets=time_offsets,
        phase_center_dec=phase_center_dec,
        telescope_latitude=telescope_latitude,
        world=world,
    )[:, :, :2].reshape(baselines.shape[0], time_offsets.size, 2)

    # grid each baseline type into uv plane
    dim = ugrid_edges.shape[1] - 1

    uvsum = np.zeros((len(frequencies), dim, dim))

    for i, freq in enumerate(frequencies):
        uvws = (proj_bls * (freq / speed_of_light)).to_value(un.dimensionless_unscaled)

        # Allow the possibility of frequency-dependent ugrid.
        if ugrid_edges.ndim == 1:
            rng = (ugrid_edges[0], ugrid_edges[-1])
        else:
            rng = (ugrid_edges[i, 0], ugrid_edges[i, -1])

        if coherent:
            uvsum[i] = histogram2d(
                uvws[:, :, 0].flatten(),
                uvws[:, :, 1].flatten(),
                range=[rng, rng],
                bins=(dim, dim),
                weights=weights,
            )
        else:
            for uvw, nbls in tqdm.tqdm(
                zip(uvws, weights, strict=False),
                desc="gridding baselines",
                unit="baselines",
                disable=not config.PROGRESS,
                total=len(weights),
            ):
                uvsum[i] += (
                    histogram2d(
                        uvw[:, 0],
                        uvw[:, 1],
                        range=[rng, rng],
                        bins=(dim, dim),
                    )
                    * nbls
                ) ** 2

            uvsum = np.sqrt(uvsum)

    return uvsum
