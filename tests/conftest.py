"""Pytest configuration file."""

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def tmpdirec(tmpdir_factory):
    return Path(tmpdir_factory.mktemp("configs"))
