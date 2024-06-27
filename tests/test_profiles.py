"""Test the profiles module."""

import astropy.units as un
import numpy as np
import pytest
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


all_profiles = [Observatory.from_profile(p) for p in get_builtin_profiles()]


def test_load_others():
    assert len({np.sum(o.antpos) for o in all_profiles}) == len(all_profiles)


def test_bad_profile():
    with pytest.raises(FileNotFoundError, match="profile invalid not available"):
        Observatory.from_profile("invalid")
