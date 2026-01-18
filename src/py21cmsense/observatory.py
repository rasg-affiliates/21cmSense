"""
Module providing the definition of an Observatory.

This replaces the original usage of an aipy.AntennaArray with something much more
simple, and suited to the needs of this particular package.
"""

from __future__ import annotations

import collections
import logging
from collections import defaultdict
from collections.abc import Callable
from functools import cached_property
from pathlib import Path

import attrs
import numpy as np
import tqdm
from astropy import constants as cnst
from astropy import units as un
from astropy.io.misc import yaml
from attr import validators as vld
from hickleable import hickleable

from . import _utils as ut
from . import beam, config
from . import units as tp

logger = logging.getLogger(__name__)

DATA = Path(__file__).parent / "data"


def get_builtin_profiles() -> list[str]:
    """Print available built-in profiles."""
    fls = (DATA / "profiles").glob("*.yaml")
    return [fl.stem for fl in fls]


@hickleable(evaluate_cached_properties=True)
@attrs.define(kw_only=True, order=False)
class Observatory:
    """
    A class defining an interferometric Observatory and its properties.

    Parameters
    ----------
    antpos : array
        An array with shape (Nants, 3) specifying the positions of the antennas.
        These should be in the ENU (East-North-Up) frame, relative to a central location
        given by `latitude`. If not a Quantity, units are assumed to be meters.
    beam : :class:`~py21cmsense.beam.PrimaryBeam` instance
        A beam, assumed to be homogeneous across antennas.
    latitude : float or Quantity, optional
        Latitude of the array center. If a float, assumed to be in radians.
        Note that longitude is not required, as we assume an isotropic sky.
    Trcv
        Receiver temperature, either a temperature Quantity, or a callable that
        takes a single frequency Quantity and returns a temperature Quantity.
    min_antpos, max_antpos
        The minimum/maximum radial distance to include antennas (from the origin
        of the array). Assumed to be in units of meters if no units are supplied.
        Can be used to limit antennas in arrays like HERA and SKA that
        have a "core" and "outriggers". The minimum is inclusive, and maximum exclusive.
    beam_crossing_time_incl_latitude
        Whether the beam-crossing time is dependent on the latitude of the telescope.
        By default it is, so that the beam-crossing time is
        ``tday * FWHM / (2pi cos(lat))``. This affects both the thermal and sample
        variance calculations.
    world: string
        A string specifying whether the telescope is on the Earth or the moon.
    """

    _antpos: tp.Length = attrs.field(eq=attrs.cmp_using(eq=np.array_equal))
    beam: beam.PrimaryBeam = attrs.field(validator=vld.instance_of(beam.PrimaryBeam))
    latitude: un.rad = attrs.field(
        default=0 * un.rad,
        validator=ut.between(-np.pi * un.rad / 2, np.pi * un.rad / 2),
    )
    Trcv: tp.Temperature | Callable = attrs.field(default=100 * un.K)
    max_antpos: tp.Length = attrs.field(
        default=np.inf * un.m, validator=(tp.vld_physical_type("length"), ut.positive)
    )
    min_antpos: tp.Length = attrs.field(
        default=0.0 * un.m, validator=(tp.vld_physical_type("length"), ut.nonnegative)
    )
    beam_crossing_time_incl_latitude: bool = attrs.field(default=True, converter=bool)
    world: str = attrs.field(default="earth", validator=vld.in_(["earth", "moon"]))

    @_antpos.validator
    def _antpos_validator(self, att, val):
        tp.vld_physical_type("length")(self, att, val)
        if val.ndim != 2:
            raise ValueError("antpos must be a 2D array.")

        if val.shape[-1] != 3:
            raise ValueError("antpos must have shape (Nants, 3).")

        if val.shape[0] <= 1:
            raise ValueError("antpos must have at least two antennas.")

    @cached_property
    def antpos(self) -> np.ndarray:
        """The positions of antennas in the array in units of metres."""
        # Mask out some antennas if a max_antpos is set in the YAML
        _n = len(self._antpos)
        sq_len = np.sum(np.square(self._antpos), axis=1)
        antpos = self._antpos[
            np.logical_and(
                sq_len >= self.min_antpos**2,
                sq_len < self.max_antpos**2,
            )
        ]

        if self.max_antpos < np.inf or self.min_antpos > 0:
            logger.info(
                f"Removed {_n - len(antpos)} antennas using given "
                f"max_antpos={self.max_antpos} m and min_antpos={self.min_antpos} m."
            )

        return antpos

    @Trcv.validator
    def _trcv_vld(self, att, val):
        if callable(val):
            try:
                y = val(1 * un.MHz)
            except Exception as e:
                raise ValueError(
                    "Trcv function must take a frequency Quantity and "
                    "return a temperature Quantity."
                ) from e

            if not (isinstance(y, un.Quantity) and y.unit.physical_type == "temperature"):
                raise ValueError("Trcv function must return a temperature Quantity.")
        else:
            tp.vld_physical_type("temperature")(self, att, val)

    @property
    def frequency(self) -> un.Quantity[un.MHz]:
        """Central frequency of the observation."""
        return self.beam.frequency.to("MHz")

    @cached_property
    def n_antennas(self) -> int:
        """Number of antennas in the array."""
        return len(self.antpos)

    def clone(self, **kwargs) -> Observatory:
        """Return a clone of this instance, but change kwargs."""
        return attrs.evolve(self, **kwargs)

    @classmethod
    def from_uvdata(cls, uvdata, beam: beam.PrimaryBeam, **kwargs) -> Observatory:
        """Instantiate an Observatory from a :class:`pyuvdata.UVData` object."""
        return cls(
            antpos=uvdata.telescope.antenna_positions,
            beam=beam,
            latitude=uvdata.telescope.location.lat,
            **kwargs,
        )

    @classmethod
    def from_yaml(cls, yaml_file: str | dict, frequency: tp.Frequency | None = None) -> Observatory:
        """Instantiate an Observatory from a compatible YAML config file."""
        if isinstance(yaml_file, (str, Path)):
            with open(yaml_file) as fl:
                lines = fl.read()
                lines = lines.replace("{{ DATA_PATH }}", str(DATA.absolute()))
                data = yaml.load(lines)
        elif isinstance(yaml_file, collections.abc.Mapping):
            data = yaml_file
        else:
            raise ValueError("yaml_file must be a string filepath or a raw dict from such a file.")

        # Mask out some antennas if a max_antpos is set in the YAML
        max_antpos = data.pop("max_antpos", np.inf * un.m)
        antpos = data.pop("antpos")
        _n = len(antpos)

        antpos = antpos[np.sum(np.square(antpos), axis=1) < max_antpos**2]

        if max_antpos < np.inf * un.m:
            logger.info(
                f"Removed {_n - len(antpos)} antennas using given max_antpos={max_antpos} m."
            )

        # If we get only East and North coords, add zeros for the UP direction.
        if antpos.shape[1] == 2:
            antpos = np.hstack((antpos, np.zeros((len(antpos), 1))))

        _beam = data.pop("beam")
        if frequency is not None:
            _beam["frequency"] = frequency

        kind = _beam.pop("class")
        _beam = getattr(beam, kind)(**_beam)

        return cls(antpos=antpos, beam=_beam, **data)

    @classmethod
    def from_profile(cls, profile: str, frequency: tp.Frequency | None = None, **kwargs):
        """Instantiate the Observatory from a builtin profile.

        Parameters
        ----------
        profile
            A string label identifying the observatory. Available built-in observatories
            can be obtained with :func:`get_builtin_profiles`. For more up-to-date SKA profiles,
            check the :func:`from_ska` method.
        frequency
            The frequency at which to specify the observatory.

        Other Parameters
        ----------------
        All other parameters passed will be passed into the initializer for the class,
        overwriting the profile.
        """
        fl = DATA / "profiles" / f"{profile}.yaml"
        if not fl.exists():
            raise FileNotFoundError(
                f"profile {profile} not available. Available profiles: {get_builtin_profiles()}"
            )

        obj = cls.from_yaml(fl, frequency=frequency)
        return obj.clone(**kwargs)

    @classmethod
    def from_ska(
        cls,
        subarray_template: str,
        Trcv: tp.Temperature | Callable = 100 * un.K,  # noqa N803
        frequency: tp.Frequency | None = 150.0 * un.MHz,
        **kwargs,
    ) -> Observatory:
        """Instantiate an SKA-Low or SKA-Mid subarray.

        Parameters
        ----------
        subarray_template
            The SKA subarray / substation template to use.
            See https://www.skao.int/en/science-users/ska-tools/543/ska-subarray-templates-library
        Trcv, optional
            Receiver temperature, either a temperature Quantity, or a callable that
            takes a single frequency Quantity and returns a temperature Quantity.
            Default is 100 K.
        frequency, optional
            The frequency at which to specify the observatory. Default is 150 MHz.

        Other Parameters
        ----------------
        All other parameters passed will be passed into get_subarray_template() from
        the ska-ost-array-config package; see the ska-ost-array-config documentation.
        """
        try:
            from ska_ost_array_config import get_subarray_template
        except ImportError as exception:  # pragma: no cover
            raise ImportError(
                "ska-ost-array-config package is required, "
                + "see https://gitlab.com/ska-telescope/ost/ska-ost-array-config"
            ) from exception

        subarray = get_subarray_template(subarray_template, **kwargs)

        antpos = subarray.array_config.xyz.data * un.m
        _beam = beam.GaussianBeam(
            frequency=frequency, dish_size=np.array(subarray.array_config.diameter)[0] * un.m
        )
        lat = subarray.array_config.location.lat.rad * un.rad
        return cls(antpos=antpos, beam=_beam, latitude=lat, Trcv=Trcv)

    @cached_property
    def baselines_metres(self) -> tp.Meters:
        """Raw baseline distances in metres for every pair of antennas.

        Shape is ``(Nant, Nant, 3)``.
        """
        # this does an "outer" subtraction, leaving the inner 2- or 3- length positions
        # as atomic quantities.
        return (self.antpos[np.newaxis, :, :] - self.antpos[:, np.newaxis, :]).to(un.m)

    def projected_baselines(
        self,
        baselines: tp.Length | None = None,
        time_offset: tp.Time = 0 * un.hour,
        phase_center_dec: tp.Angle | None = None,
        in_wavelengths: bool = True,
    ) -> np.ndarray:
        """Compute *projected* baseline vectors.

        Phased to a point that has rotated off zenith by some time_offset.

        Parameters
        ----------
        baselines
            The baseline co-ordinates to project, assumed to be in metres.
            If not provided, uses all baselines of the observatory.
            Shape of the array can be (N,N,3) or (N, 3).
            The co-ordinates are expected to be in ENU.
        time_offset
            The amount of time elapsed since the phase center was at zenith.
            Assumed to be in days unless otherwise defined. May be negative.
        phase_center_dec
            The declination of the phase center of the observation. By default, the
            same as the latitude of the array.

        Returns
        -------
        An array the same shape as :attr:`baselines_metres`, but phased to the
        new phase centre.
        """
        if baselines is None:
            baselines = self.baselines_metres

        bls = ut.project_baselines(
            baselines=baselines,
            telescope_latitude=self.latitude,
            time_offsets=time_offset,
            phase_center_dec=phase_center_dec,
            world=self.world,
        )

        if in_wavelengths:
            return bls * self.metres_to_wavelengths
        else:
            return bls

    @cached_property
    def metres_to_wavelengths(self) -> un.Quantity[1 / un.m]:
        """Conversion factor for metres to wavelengths at fiducial frequency."""
        return (self.frequency / cnst.c).to("1/m")

    @cached_property
    def baseline_lengths(self) -> np.ndarray:
        """Lengths of baselines in units of wavelengths, shape (Nant, Nant)."""
        return np.sqrt(np.sum(self.projected_baselines() ** 2, axis=-1))

    @cached_property
    def shortest_baseline(self) -> float:
        """Shortest baseline in units of wavelengths."""
        return np.min(self.baseline_lengths[self.baseline_lengths > 0])

    @cached_property
    def longest_baseline(self) -> float:
        """Longest baseline in units of wavelengths."""
        return np.max(self.baseline_lengths)

    @cached_property
    def observation_duration(self) -> un.Quantity[un.day]:
        """The time it takes for the sky to drift through the FWHM."""
        latfac = np.cos(self.latitude) if self.beam_crossing_time_incl_latitude else 1
        if self.world == "earth":
            return un.day * self.beam.fwhm / (2 * np.pi * un.rad * latfac)
        else:
            return 27.3 * un.day * self.beam.fwhm / (2 * np.pi * un.rad * latfac)

    def get_redundant_baselines(
        self,
        baseline_filters: Callable | tuple[Callable] = (),
        ndecimals: int = 1,
        add_conjugates: bool = True,
    ) -> dict[tuple[float, float, float], list[tuple[int, int]]]:
        """
        Determine all baseline groups.

        Parameters
        ----------
        baseline_filters
            Callable function (or functions) of a single 3-coordinate baseline vector
            that returns a bool indicating whether to include the baseline.
        ndecimals : int, optional
            The number of decimals to which the UV points must be the same to be
            considered redundant.

        Returns
        -------
        dict: a dictionary in which keys are 3-tuples of ``(u,v, |u|)`` co-ordinates and
            values are lists of 2-tuples, where each 2-tuple consists of the indices
            of a pair of antennas with those co-ordinates.
        """
        uvbins = defaultdict(list)
        baseline_filters = tp._tuplify(baseline_filters, 1)

        def filt(blm):
            return all(filt(blm) for filt in baseline_filters)

        # Everything here is in wavelengths
        uvw = self.projected_baselines()[:, :, :2].value
        uvw = np.round(uvw, decimals=ndecimals)
        bl_lens = np.round(self.baseline_lengths.value, decimals=ndecimals)

        # group redundant baselines
        for i in tqdm.tqdm(
            range(self.n_antennas - 1),
            desc="finding redundancies",
            unit="ants",
            disable=not config.PROGRESS,
        ):
            for j in range(i + 1, self.n_antennas):
                blm = self.baselines_metres[i, j]

                # Check if we want to include this baseline.
                if not filt(blm):
                    continue

                bl_len = bl_lens[i, j]  # in wavelengths

                u, v = uvw[i, j]

                # add the uv point and its inverse to the redundant baseline dict.
                uvbins[(u, v, bl_len)].append((i, j))

        if add_conjugates:
            for (u, v, bl_len), antpairs in list(uvbins.items()):
                uvbins[(-u, -v, bl_len)] = [(j, i) for (i, j) in antpairs]

        return uvbins

    def time_offsets_from_obs_int_time(
        self, integration_time: tp.Time, observation_duration: tp.Time | None = None
    ):
        """Compute a list of time offsets within an LST-bin.

        The LSTs 'within a bin' are added coherently for a given baseline group.
        Time offsets are with respect to an arbitrary time, and describe the rotation of
        a hypothetical point through zenith.

        Parameters
        ----------
        integration_time
            Time for single snapshot.
        observation_duration
            Duration of the LST bin (for single night).

        Returns
        -------
        array :
            Time offsets (in julian days).
        """
        if observation_duration is None:
            observation_duration = self.observation_duration

        assert integration_time <= observation_duration

        return (
            np.arange(
                -observation_duration.to("day").value / 2,
                observation_duration.to("day").value / 2,
                integration_time.to("day").value,
            )
            << un.day
        )

    def baseline_coords_from_groups(self, baseline_groups) -> un.Quantity[un.m]:
        """Convert a dictionary of baseline groups to an array of ENU co-ordinates."""
        out = np.zeros((len(baseline_groups), 3)) * un.m
        for i, antpairs in enumerate(baseline_groups.values()):
            out[i] = self.baselines_metres[antpairs[0][0], antpairs[0][1]]
        return out

    @staticmethod
    def baseline_weights_from_groups(baseline_groups) -> np.ndarray:
        """Get number of baselines in each group.

        Parameters
        ----------
        baseline_groups
            A dictionary in the format output by :func:`get_redundant_baselines`.

        Returns
        -------
        weights
            An array containing the number of baselines in each group.
        """
        return np.array([len(antpairs) for antpairs in baseline_groups.values()])

    def grid_baselines(
        self,
        coherent: bool,
        baselines: tp.Length | None = None,
        weights: np.ndarray | None = None,
        integration_time: tp.Time = 60.0 * un.s,
        baseline_filters: Callable | tuple[Callable] = (),
        observation_duration: tp.Time | None = None,
        ndecimals: int = 1,
        phase_center_dec: tp.Angle | None = None,
        max_chunk_mem_gb: float = 1.0,
    ) -> np.ndarray:
        """
        Grid baselines onto a pre-determined uvgrid, accounting for earth rotation.

        Parameters
        ----------
        baselines : array_like, optional
            The baseline co-ordinates to project, assumed to be in metres.
            If not provided, calculates effective baselines by finding redundancies on
            all baselines in the observatory. Shape of the array can be (N,N,3) or (N, 3).
            The co-ordinates are expected to be in ENU. If `baselines` is provided,
            `weights` must also be provided.
        weights: array_like, optional
            An array of the same length as `baselines`, giving the number of independent
            baselines at each co-ordinate. If not provided, calculates effective
            baselines by finding redundancies on all baselines in the observatory.
            If `baselines` is provided, `weights` must also be provided.
        integration_time : float or Quantity, optional
            The amount of time integrated into a snapshot visibility, assumed
            to be in seconds.
        baseline_filters
            A function that takes a single value: a length-3 array of baseline co-ordinates,
            and returns a bool indicating whether to include the baseline. Built-in filters
            are provided in the :mod:`~baseline_filters` module.
        observation_duration : float or Quantity, optional
            Amount of time in a single (coherent) LST bin, assumed to be in minutes.
        ndecimals : int, optional
            Number of decimals to which baselines must match to be considered redundant.

        Returns
        -------
        array :
            Shape [n_baseline_groups, Nuv, Nuv]. The coherent sum of baselines within
            grid cells given by :attr:`ugrid`. One can treat different baseline groups
            independently, or sum over them.

        See Also
        --------
        grid_baselines_coherent :
            Coherent sum over baseline groups of the output of this method.
        grid_baseline_incoherent :
            Incoherent sum over baseline groups of the output of this method.
        """
        if baselines is None:
            baseline_groups = self.get_redundant_baselines(
                baseline_filters=baseline_filters, ndecimals=ndecimals
            )
            baselines = self.baseline_coords_from_groups(baseline_groups)
            weights = self.baseline_weights_from_groups(baseline_groups)

        assert un.get_physical_type(baselines) == "length"
        assert baselines.ndim in (2, 3)
        assert un.get_physical_type(integration_time) == "time"
        if observation_duration is not None:
            assert un.get_physical_type(observation_duration) == "time"

        if weights is None:
            raise ValueError("If baselines are provided, weights must also be provided.")

        time_offsets = self.time_offsets_from_obs_int_time(integration_time, observation_duration)

        bl_max = np.sqrt(np.max(np.sum(baselines**2, axis=1)))

        return ut.grid_baselines(
            coherent=coherent,
            baselines=baselines,
            weights=weights,
            time_offsets=time_offsets,
            frequencies=[self.beam.frequency],
            ugrid_edges=self.ugrid_edges(bl_max),
            phase_center_dec=phase_center_dec,
            telescope_latitude=self.latitude,
            world=self.world,
            max_chunk_mem_gb=max_chunk_mem_gb,
        )[0]

    def longest_used_baseline(self, bl_max: tp.Length = np.inf * un.m) -> float:
        """Determine the maximum baseline length kept in the array, in wavelengths."""
        if np.isinf(bl_max):
            return self.longest_baseline

        # Note we don't do the conversion in-place!
        bl_max = bl_max * self.metres_to_wavelengths
        return np.max(self.baseline_lengths[self.baseline_lengths <= bl_max])

    def ugrid_edges(self, bl_max: tp.Length = np.inf * un.m) -> np.ndarray:
        """Get a uv grid out to the maximum used baseline smaller than given bl_max.

        The resulting array represents the *edges* of the grid (so the number of cells
        is one fewer than this).

        Parameters
        ----------
        bl_max : float or Quantity
            Include all baselines smaller than this number. Units of m.

        Returns
        -------
        array :
            1D array of regularly spaced un.
        """
        bl_max = self.longest_used_baseline(bl_max)

        # We're doing edges of bins here, and the first edge is at uv_res/2
        n_positive = int(np.ceil((bl_max - self.beam.uv_resolution / 2) / self.beam.uv_resolution))

        # Grid from uv_res/2 to just past (or equal to) bl_max, in steps of resolution.
        positive = np.linspace(
            self.beam.uv_resolution / 2,
            self.beam.uv_resolution / 2 + n_positive * self.beam.uv_resolution,
            n_positive + 1,
        )
        return np.concatenate((-positive[::-1], positive))

    def ugrid(self, bl_max: tp.Length = np.inf * un.m) -> np.ndarray:
        """Centres of the UV grid plane."""
        # Shift the edges by half a cell, and omit the last one
        edges = self.ugrid_edges(bl_max)
        return (edges[1:] + edges[:-1]) / 2
