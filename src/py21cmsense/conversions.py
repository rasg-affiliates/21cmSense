"""
Common 21 cm conversions.

Provides conversions between observing co-ordinates and cosmological co-ordinates.
"""

from __future__ import annotations

import numpy as np
from astropy import constants as cnst
from astropy import units as un
from astropy.cosmology import FLRW, Planck15
from astropy.cosmology.units import littleh

from . import units as tp

# The frequency of the 21cm line emission.
f21 = 1.42040575177 * un.GHz


def f2z(fq: tp.Frequency) -> float:
    """
    Convert frequency to redshift for 21 cm line.

    Parameters
    ----------
    fq
        The frequency or frequencies of observation.

    Returns
    -------
    dimensionless astropy.Quantity : The redshift
    """
    return f21 / fq - 1


def z2f(z: float | np.array) -> un.Quantity[un.GHz]:
    """
    Convert redshift to z=0 frequency for 21 cm line.

    Parameters
    ----------
    z
        Redshift

    Returns
    -------
    astropy.Quantity : the frequency
    """
    return f21 / (1 + z)


def dL_dth(
    z: float | np.array, cosmo: FLRW = Planck15, approximate=False, with_h: bool = True
) -> un.Quantity[un.Mpc / un.rad / littleh]:
    """
    Return the factor to convert radians to transverse distance at redshift z.

    Parameters
    ----------
    z : float
        The redshift

    Returns
    -------
    astropy.Quantity : the factor (in Mpc/h/radian) which converts from an angle
        to a transverse distance.

    Notes
    -----
    From Furlanetto et al. (2006)
    """
    if approximate:
        fac = littleh if with_h else cosmo.h
        return (1.9 * (1.0 / un.arcmin) * ((1 + z) / 10.0) ** 0.2).to(1 / un.rad) * un.Mpc / fac
    else:
        fac = cosmo.h / littleh if with_h else 1.0
        return fac * cosmo.comoving_transverse_distance(z) / un.rad


def dL_df(
    z: float | np.array,
    cosmo: FLRW = Planck15,
    approximate=False,
    with_h: bool = True,
) -> un.Quantity[un.Mpc / un.MHz / littleh]:
    """
    Get the factor to convert bandwidth to line-of-sight distance in Mpc/h.

    Parameters
    ----------
    z : float
        The redshift
    """
    if approximate:
        fac = littleh if with_h else cosmo.h
        return (
            (1.7 / 0.1)
            * ((1 + z) / 10.0) ** 0.5
            * (cosmo.Om0 / 0.15) ** -0.5
            * un.Mpc
            / fac
            / un.MHz
        )
    else:
        fac = cosmo.h / littleh if with_h else 1.0
        unit = "Mpc/(MHz*littleh)" if with_h else "Mpc/MHz"
        return (fac * cnst.c * (1 + z) / (z2f(z) * cosmo.H(z))).to(unit)


def dk_du(
    z: float | np.array,
    cosmo: FLRW = Planck15,
    approximate=False,
    with_h: bool = True,
) -> un.Quantity[littleh / un.Mpc]:
    """
    Get factor converting bl length in wavelengths to h/Mpc.

    Parameters
    ----------
    z : float
        redshift

    Notes
    -----
    Valid for u >> 1
    """
    # from du = 1/dth, which derives from du = d(sin(th)) using the small-angle approx
    return 2 * np.pi / dL_dth(z, cosmo, approximate=approximate, with_h=with_h) / un.rad


def dk_deta(
    z: float | np.array,
    cosmo: FLRW = Planck15,
    approximate=False,
    with_h: bool = True,
) -> un.Quantity[un.MHz * littleh / un.Mpc]:
    """
    Get gactor converting inverse frequency to inverse distance.

    Parameters
    ----------
    z: float
        Redshift
    """
    return 2 * np.pi / dL_df(z, cosmo, approximate=approximate, with_h=with_h)


def X2Y(
    z: float | np.array, cosmo: FLRW = Planck15, approximate=False, with_h: bool = True
) -> un.Quantity[un.Mpc**3 / littleh**3 / un.steradian / un.MHz]:
    """
    Obtain the conversion factor between observing co-ordinates and cosmological volume.

    Parameters
    ----------
    z: float
        Redshift
    cosmo: astropy.cosmology.FLRW instance
        A cosmology.

    Returns
    -------
    astropy.Quantity: the conversion factor. Units are Mpc^3/h^3 / (sr MHz).
    """
    return dL_dth(z, cosmo, approximate=approximate, with_h=with_h) ** 2 * dL_df(
        z, cosmo, approximate=approximate, with_h=with_h
    )
