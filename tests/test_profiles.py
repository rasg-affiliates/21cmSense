import pytest

import astropy.units as un

from py21cmsense.observatory import Observatory, get_builtin_profiles


def test_get_available():
    avail = get_builtin_profiles()
    assert "HERA-H1C-IDR3" in avail


def test_load_hera_h1c():
    hera = Observatory.from_profile("HERA-H1C-IDR3")
    assert hera.Trcv == 100 * un.K
    assert hera.antpos.shape == (71, 3)


def test_load_hera_with_custom():
    hera = Observatory.from_profile("HERA-H1C-IDR3", Trcv=200 * un.K)
    assert hera.Trcv == 200 * un.K
