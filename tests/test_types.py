import pytest

import attr
from astropy import units as u

from py21cmsense import units as tp


def test_vld_physical_type():
    @attr.s
    class K:
        a = attr.ib(validator=tp.vld_physical_type("frequency"))

    with pytest.raises(tp.UnitError, match="a must be an astropy Quantity!"):
        K(1)

    with pytest.raises(
        u.UnitConversionError,
        match="a must have physical type of 'frequency'. Got 'time'",
    ):
        K(1 * u.s)

    assert K(1 * u.MHz).a.unit.physical_type == "frequency"


def test_vld_unit():
    @attr.s
    class K:
        a = attr.ib(validator=tp.vld_unit(u.GHz))

    with pytest.raises(tp.UnitError, match="a must be an astropy Quantity!"):
        K(1)

    with pytest.raises(u.UnitConversionError, match="a not convertible to GHz. Got s"):
        K(1 * u.s)

    assert K(1 * u.MHz).a.unit.physical_type == "frequency"


def test_tuplify():
    x = tp._tuplify(1)
    assert x == (1,)

    x = tp._tuplify([1])
    assert x == (1,)

    x = tp._tuplify((1,))
    assert x == (1,)

    x = tp._tuplify((1, 2))
    assert x == (1, 2)

    x = tp._tuplify(None, n=3)
    assert x == (None, None, None)
