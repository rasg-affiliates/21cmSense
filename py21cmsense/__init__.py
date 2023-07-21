"""A package for calculate sensitivies of 21-cm interferometers."""
from pkg_resources import DistributionNotFound, get_distribution

try:
    __version__ = get_distribution("21cmSense").version
except DistributionNotFound:
    __version__ = "unknown"
finally:
    del get_distribution, DistributionNotFound

from . import yaml
from .antpos import hera
from .beam import GaussianBeam
from .observation import Observation
from .observatory import Observatory
from .sensitivity import PowerSpectrum
