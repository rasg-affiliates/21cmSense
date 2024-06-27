"""Test the yaml module."""

import pickle

import numpy as np
import pytest
from astropy import units as un
from astropy.io.misc import yaml
from py21cmsense.yaml import LoadError


def test_file_not_found():
    with pytest.raises(IOError, match="not found"):
        yaml.load("!txt non-existent-file.txt")


def test_bad_loader(tmpdirec):
    fl = tmpdirec / "test-npy"

    with open(fl, "w") as ff:
        ff.write("some-text")

    with pytest.raises(LoadError):
        yaml.load(f"!npy {fl}")


def test_pickle_loader(tmpdirec):
    pkl = tmpdirec / "test-pkl.pkl"

    obj = {"an": "object"}
    with open(pkl, "wb") as fl:
        pickle.dump(obj, fl)

    d = yaml.load(f"!pkl {pkl}")

    assert d == obj


def test_npz_loader(tmpdirec):
    npz = tmpdirec / "test-npz.npz"

    obj = {"an": np.linspace(0, 1, 10), "b": np.zeros(10)}

    np.savez(npz, **obj)

    d = yaml.load(f"!npz {npz}")

    for k, v in d.items():
        assert k in obj
        assert np.allclose(v, obj[k])


def test_txt_loader_with_unit(tmpdirec):
    txt = tmpdirec / "test-txt.txt"

    obj = np.linspace(0, 1, 10)

    np.savetxt(txt, obj)

    d = yaml.load(f"!txt {txt} | m")
    assert d.unit == un.m
    assert np.allclose(d, obj * un.m)
