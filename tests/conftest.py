"""Pytest configuration file."""

from pathlib import Path

import pytest
from astropy.utils import iers


@pytest.fixture(scope="session", autouse=True)
def no_iers_download():
    """Use the IERS tables bundled with astropy instead of downloading them.

    The sub-millisecond UT1-UTC corrections in the auto-downloaded table make no
    difference to any sensitivity calculation here, and downloading them makes the
    test suite depend on the IERS datacenter being up (which, under ``-W error``,
    turns an outage into a few dozen test failures).
    """
    with iers.conf.set_temp("auto_download", False), iers.conf.set_temp("auto_max_age", None):
        yield


@pytest.fixture(scope="session")
def tmpdirec(tmpdir_factory):
    return Path(tmpdir_factory.mktemp("configs"))
