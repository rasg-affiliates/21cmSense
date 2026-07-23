"""Simplistic beam definitions."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod

import attrs
from astropy import constants as cnst
from astropy import units as un
from hickleable import hickleable

from . import _utils as ut
from . import units as tp


@hickleable(evaluate_cached_properties=True)
@attrs.define(frozen=True)
class PrimaryBeam(metaclass=ABCMeta):
    """A Base class defining a Primary Beam and the methods it requires to define.

    Required methods on subclasses are :meth:`area`, :meth:`width`, :meth:`first_null`,
    :meth:`sq_area` and :meth:`uv_resolution`..

    Note that 21cmSense currently only uses the beam width as a means to calculate
    the beam-crossing time, so precise shape is not very important. For that reason,
    it is not very important to implement beam sub-classes.
    """

    def new(self, **kwargs) -> PrimaryBeam:
        """Return a clone of this instance, but change kwargs."""
        return attrs.evolve(self, **kwargs)

    @abstractmethod
    def area(self, frequency: tp.Frequency) -> un.Quantity[un.steradian]:
        """Beam area [units: sr]."""

    @abstractmethod
    def width(self, frequency: tp.Frequency) -> un.Quantity[un.radians]:
        """Beam width [units: rad]."""

    @abstractmethod
    def first_null(self, frequency: tp.Frequency) -> un.Quantity[un.radians]:
        """Compute the zenith angle of the first null of the beam."""

    @abstractmethod
    def sq_area(self, frequency: tp.Frequency) -> un.Quantity[un.steradian]:
        """Compute the area of the beam^2."""

    def b_eff(self, frequency: tp.Frequency) -> un.Quantity[un.steradian]:
        r"""Get the effective beam area (Parsons 2014).

        Defined as :math:`(\int B(\Omega) d \Omega)^2 / \int B^2 d\Omega`.
        """
        return self.area(frequency) ** 2 / self.sq_area(frequency)

    @abstractmethod
    def uv_resolution(self, frequency: tp.Frequency) -> un.Quantity[1 / un.radians]:
        """Compute the UV footprint of the beam."""

    @classmethod
    def from_uvbeam(cls) -> PrimaryBeam:
        """Generate the beam object from a :class:`pyuvdata.UVBeam` object."""
        raise NotImplementedError()


@attrs.define(frozen=True)
class GaussianBeam(PrimaryBeam):
    """
    A simple Gaussian Primary beam.

    Parameters
    ----------
    frequency
        The fiducial frequency at which the beam operates, assumed to be in MHz
        unless otherwise defined.
    dish_size
        The size of the (assumed circular) dish, assumed to be in meters unless
        otherwise defined. This generates the beam size.
    """

    dish_size: tp.Length = attrs.field(validator=(tp.vld_physical_type("length"), ut.positive))

    def dish_size_in_lambda(self, frequency: tp.Frequency) -> float:
        """Compute the dish size in units of wavelengths."""
        return (self.dish_size * frequency / cnst.c).to("").value

    def uv_resolution(self, frequency: tp.Frequency) -> un.Quantity[1 / un.radian]:
        """Compute the appropriate resolution of a UV cell given the beam size."""
        return self.dish_size_in_lambda(frequency)

    def area(self, frequency: tp.Frequency) -> un.Quantity[un.steradian]:
        """Compute the integral of the beam over angle, in sr."""
        return 1.13 * self.fwhm(frequency) ** 2

    def width(self, frequency: tp.Frequency) -> un.Quantity[un.radian]:
        """Compute the width of the beam (i.e. sigma), in radians."""
        return un.rad * 0.45 / self.dish_size_in_lambda(frequency)

    def fwhm(self, frequency: tp.Frequency) -> un.Quantity[un.radians]:
        """Compute the full-width half maximum of the beam."""
        return 2.35 * self.width(frequency)

    def sq_area(self, frequency: tp.Frequency) -> un.Quantity[un.steradian]:
        """Compute the integral of the squared beam, in sr.

        If frequency is not given, uses the instance's `frequency`
        """
        return self.area(frequency) / 2

    def first_null(self, frequency: tp.Frequency) -> un.Quantity[un.radians]:
        """Get the zenith angle of the first null of the beam.

        .. note:: The Gaussian beam has no null, and in this case we use the first null
                  for an airy disk.
        """
        return un.rad * 1.22 / self.dish_size_in_lambda(frequency)

    @classmethod
    def from_uvbeam(cls):
        """Construct the beam from a :class:`pyuvdata.UVBeam` object."""
        raise NotImplementedError("Coming Soon!")

    def clone(self, **kwargs):
        """Create a new beam with updated parameters."""
        return attrs.evolve(self, **kwargs)
