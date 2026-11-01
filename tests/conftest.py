"""Pytest configuration file."""

from pathlib import Path

import pytest
from astropy.utils import iers

# Use the IERS tables bundled with astropy (via astropy-iers-data) rather than
# downloading the latest IERS-A table from datacenter.iers.org. The download makes
# the test suite depend on a remote server, and a failed download raises an
# IERSWarning that is turned into an error when running with ``-W error``.
# Since the bundled table may be older than ``auto_max_age``, also allow
# interpolating with its (stale) predictive values instead of raising an error;
# sub-arcsecond Earth-orientation precision is irrelevant for the tests.
iers.conf.auto_download = False
iers.conf.auto_max_age = None


@pytest.fixture(scope="session")
def tmpdirec(tmpdir_factory):
    return Path(tmpdir_factory.mktemp("configs"))
