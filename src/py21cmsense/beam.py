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

    frequency: tp.Frequency = attrs.field(
        validator=(tp.vld_physical_type("frequency"), ut.positive),
    )

    def new(self, **kwargs) -> PrimaryBeam:
        """Return a clone of this instance, but change kwargs."""
        return attrs.evolve(self, **kwargs)

    def at(self, frequency: tp.Frequency) -> PrimaryBeam:
        """Get a copy of the object at a new frequency."""
        return attrs.evolve(self, frequency=frequency)

    @property
    @abstractmethod
    def area(self) -> un.Quantity[un.steradian]:
        """Beam area [units: sr]."""

    @property
    @abstractmethod
    def width(self) -> un.Quantity[un.radians]:
        """Beam width [units: rad]."""

    @property
    @abstractmethod
    def first_null(self) -> un.Quantity[un.radians]:
        """An approximation of the first null of the beam."""

    @property
    @abstractmethod
    def sq_area(self) -> un.Quantity[un.steradian]:
        """The area of the beam^2."""

    @property
    def b_eff(self) -> un.Quantity[un.steradian]:
        r"""Get the effective beam area (Parsons 2014).

        Defined as :math:`(\int B(\Omega) d \Omega)^2 / \int B^2 d\Omega`.
        """
        return self.area**2 / self.sq_area

    @property
    @abstractmethod
    def uv_resolution(self) -> un.Quantity[1 / un.radians]:
        """The UV footprint of the beam."""

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

    @property
    def wavelength(self) -> un.Quantity[un.m]:
        """The wavelength of the observation."""
        return (cnst.c / self.frequency).to("m")

    @property
    def dish_size_in_lambda(self) -> float:
        """The dish size in units of wavelengths."""
        return (self.dish_size / (cnst.c / self.frequency)).to("").value

    @property
    def uv_resolution(self) -> un.Quantity[1 / un.radian]:
        """The appropriate resolution of a UV cell given the beam size."""
        return self.dish_size_in_lambda

    @property
    def area(self) -> un.Quantity[un.steradian]:
        """The integral of the beam over angle, in sr."""
        return 1.13 * self.fwhm**2

    @property
    def width(self) -> un.Quantity[un.radian]:
        """The width of the beam (i.e. sigma), in radians."""
        return un.rad * 0.45 / self.dish_size_in_lambda

    @property
    def fwhm(self) -> un.Quantity[un.radians]:
        """The full-width half maximum of the beam."""
        return 2.35 * self.width

    @property
    def sq_area(self) -> un.Quantity[un.steradian]:
        """The integral of the squared beam, in sr.

        If frequency is not given, uses the instance's `frequency`
        """
        return self.area / 2

    @property
    def first_null(self) -> un.Quantity[un.radians]:
        """The angle of the first null of the beam.

        .. note:: The Gaussian beam has no null, and in this case we use the first null
                  for an airy disk.
        """
        return un.rad * 1.22 / self.dish_size_in_lambda

    @classmethod
    def from_uvbeam(cls):
        """Construct the beam from a :class:`pyuvdata.UVBeam` object."""
        raise NotImplementedError("Coming Soon!")

    def clone(self, **kwargs):
        """Create a new beam with updated parameters."""
        return attrs.evolve(self, **kwargs)
