"""Module dealing with types and units throughout the package."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import attrs
from astropy import constants as cnst
from astropy import units as un
from astropy.cosmology.units import littleh, redshift

un.add_enabled_units([littleh, redshift])


class UnitError(ValueError):
    """An error pertaining to having incorrect units."""


Length = un.Quantity["length"]
Meters = un.Quantity["m"]
Time = un.Quantity["time"]
Frequency = un.Quantity["frequency"]
Temperature = un.Quantity["temperature"]
TempSquared = un.Quantity[un.get_physical_type("temperature") ** 2]
Wavenumber = un.Quantity[littleh / un.Mpc]
Delta = un.Quantity[un.mK**2]
Angle = un.Quantity["angle"]

time_as_distance = [
    (
        un.s,
        un.m,
        lambda x: cnst.c.to_value("m/s") * x,
        lambda x: x / cnst.c.to_value("m/s"),
    )
]


def vld_physical_type(unit: str) -> Callable[[Any, attrs.Attribute, Any], None]:
    """Attr validator to check physical type."""

    def _check_type(self: Any, att: attrs.Attribute, val: Any):
        if not isinstance(val, un.Quantity):
            raise UnitError(f"{att.name} must be an astropy Quantity!")
        if val.unit.physical_type != unit:
            raise un.UnitConversionError(
                f"{att.name} must have physical type of '{unit}'. Got '{val.unit.physical_type}'"
            )

    return _check_type


def vld_unit(unit, equivalencies=()):
    """Attr validator to check unit equivalence."""

    def _check_unit(self, att, val):
        if not isinstance(val, un.Quantity):
            raise UnitError(f"{att.name} must be an astropy Quantity!")

        if not val.unit.is_equivalent(unit, equivalencies):
            raise un.UnitConversionError(f"{att.name} not convertible to {unit}. Got {val.unit}")

    return _check_unit


def _tuplify(x, n: int = 1):
    try:
        return tuple(x)
    except TypeError:
        return (x,) * n
