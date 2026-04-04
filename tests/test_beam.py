"""Test the beam module."""

import numpy as np
import pytest
from astropy import units

from py21cmsense import GaussianBeam, beam


def test_abc():
    with pytest.raises(TypeError):
        beam.PrimaryBeam(150.0)


def test_gaussian_beam():
    bm = GaussianBeam(dish_size=14 * units.m)

    assert bm.dish_size == 14 * units.m

    bm = GaussianBeam(dish_size=1400 * units.cm)

    assert bm.dish_size == 14 * units.m

    assert not hasattr(bm.dish_size_in_lambda, "unit")

    with pytest.raises(NotImplementedError):
        GaussianBeam.from_uvbeam()

    freq = 150 * units.MHz
    assert bm.uv_resolution(freq) == bm.dish_size_in_lambda(freq)
    assert bm.sq_area(freq) < bm.area(freq)
    assert bm.fwhm(freq) > bm.width(freq)
    assert bm.first_null(freq) < np.pi * units.rad / 2

    assert bm.new() == bm
