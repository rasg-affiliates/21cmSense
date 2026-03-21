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
@attrs.define(kw_only=True, order=False, slots=False)
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
    baseline_filters: tuple[Callable[[tp.Length], bool]] = attrs.field(
        default=(), converter=tp._tuplify
    )
    redundancy_tol: int = attrs.field(default=1, converter=int, validator=ut.nonnegative)

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
    def from_yaml(cls, yaml_file: str | dict) -> Observatory:
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

        kind = _beam.pop("class")
        _beam = getattr(beam, kind)(**_beam)

        return cls(antpos=antpos, beam=_beam, **data)

    @classmethod
    def from_profile(cls, profile: str, **kwargs):
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

        obj = cls.from_yaml(fl)
        return obj.clone(**kwargs)

    @classmethod
    def from_ska(
        cls,
        subarray_template: str,
        Trcv: tp.Temperature | Callable = 100 * un.K,  # noqa N803
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
        _beam = beam.GaussianBeam(dish_size=np.array(subarray.array_config.diameter)[0] * un.m)
        lat = subarray.array_config.location.lat.rad * un.rad
        return cls(antpos=antpos, beam=_beam, latitude=lat, Trcv=Trcv)

    @property
    def unfiltered_baselines(self) -> tp.Meters:
        """Raw baseline distances in metres for every pair of antennas.

        This includes only one of the two conjugate baselines, no autocorrelations, and
        does not apply any baseline filters.

        Shape is ``(Nant*(Nant-1)/2, 3)``.
        """
        # sort along Northing so that when we take the upper triangle
        # we get only positive y-values.
        inds = np.argsort(self.antpos[:, 1])
        antpos = self.antpos[inds]
        n = len(self.antpos)
        allbls = (antpos[np.newaxis, :, :] - antpos[:, np.newaxis, :]).to(un.m)
        indices = np.triu_indices(n, k=1)
        return allbls[indices]

    @cached_property
    def baselines(self) -> tp.Meters:
        """Baseline vectors in metres for every pair of antennas, after applying filters.

        Shape is ``(Nbls, 3)``.
        """
        bls = self.unfiltered_baselines

        # Apply baseline filters
        mask = np.ones(bls.shape[:-1], dtype=bool)
        for filt in self.baseline_filters:
            mask &= filt(bls)

        return bls[mask]

    @property
    def baselines_metres(self) -> tp.Meters:
        """Raw baseline distances in metres for every pair of antennas.

        Deprecated -- use :attr:`baselines` instead.

        Shape is ``(Nant, Nant, 3)``.
        """
        return self.baselines.to(un.m)

    def projected_baselines(
        self,
        baselines: tp.Length | None = None,
        time_offset: tp.Time = 0 * un.hour,
        phase_center_dec: tp.Angle | None = None,
    ) -> tp.Meters:
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
        An array the same shape as :attr:`baselines`, but phased to the
        new phase centre.
        """
        if baselines is None:
            baselines = self.baselines

        return ut.project_baselines(
            baselines=baselines,
            telescope_latitude=self.latitude,
            time_offsets=time_offset,
            phase_center_dec=phase_center_dec,
            world=self.world,
        )

    @cached_property
    def baseline_lengths(self) -> tp.Meters:
        """Lengths of baselines in units of meters, shape (Nbls,)."""
        return np.sqrt(np.sum(self.projected_baselines() ** 2, axis=-1))

    @cached_property
    def bl_min(self) -> un.Quantity[un.m]:
        """Shortest included baseline."""
        return self.baseline_group_lengths.min()

    @cached_property
    def bl_max(self) -> un.Quantity[un.m]:
        """Shortest included baseline."""
        return self.baseline_group_lengths.max()

    def get_redundant_baselines(self) -> list[list[int]]:
        """
        Determine all baseline groups.

        Returns
        -------
        dict: a dictionary in which keys are 3-tuples of ``(u,v, |u|)`` co-ordinates and
            values are lists of 2-tuples, where each 2-tuple consists of the indices
            of a pair of antennas with those co-ordinates.
        """
        uvbins = defaultdict(list)

        # Everything here is in meters
        uvw = self.projected_baselines()[:, :2].value
        uvw = np.round(uvw, decimals=self.redundancy_tol)

        # group redundant baselines
        for i, (u, v) in tqdm.tqdm(
            enumerate(uvw),
            desc="finding redundancies",
            unit="ants",
            disable=not config.PROGRESS,
        ):
            uvbins[(u, v)].append(i)

        return list(uvbins.values())
        return self.redundant_baseline_groups

    @cached_property
    def redundant_baseline_groups(self) -> list[list[int]]:
        """A list of lists of redundant baseline groups as indices into :attr:`baselines`."""
        return self.get_redundant_baselines()

    @cached_property
    def redundant_baseline_vectors(self) -> tp.Meters:
        """Get the baseline vectors for each redundant baseline group."""
        idx = [bls[0] for bls in self.redundant_baseline_groups]
        return self.baselines[idx]

    @cached_property
    def redundant_baseline_weights(self) -> np.ndarray:
        """Get the weights for each redundant baseline group."""
        return np.array([len(bls) for bls in self.redundant_baseline_groups])

    @cached_property
    def baseline_group_lengths(self) -> un.Quantity[un.m]:
        """The displacement magnitude of the baseline groups."""
        return np.sqrt(np.sum(self.redundant_baseline_vectors**2, axis=-1))

    def time_offsets_from_obs_int_time(
        self,
        integration_time: tp.Time,
        observation_duration: tp.Time,
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
        assert integration_time <= observation_duration

        return (
            np.arange(
                -observation_duration.to("day").value / 2,
                observation_duration.to("day").value / 2,
                integration_time.to("day").value,
            )
            << un.day
        )

    @cached_property
    def xgrid_edges(self) -> tp.Meters:
        """Get a grid of bl length to the max used baseline smaller than given bl_max.

        This grid is optimized such that the grid size is the same as the dish size,
        i.e. such that the pixels are roughly uncorrelated.

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
        bl_max = self.bl_max

        # We're doing edges of bins here, and the first edge is at uv_res/2
        n_positive = int(np.ceil((bl_max - self.beam.dish_size / 2) / self.beam.dish_size))

        # Grid from uv_res/2 to just past (or equal to) bl_max, in steps of resolution.
        xmin = self.beam.dish_size.to_value("m") / 2
        dx = self.beam.dish_size.to_value("m")
        positive = np.linspace(xmin, xmin + n_positive * dx, n_positive + 1)
        return np.concatenate((-positive[::-1], positive)) * un.m

    @property
    def xgrid(self) -> tp.Meters:
        """Get a grid of bl length to the max used baseline smaller than given bl_max.

        This grid is optimized such that the grid size is the same as the dish size,
        i.e. such that the pixels are roughly uncorrelated.

        The resulting array represents the *centers* of the grid cells.

        Parameters
        ----------
        bl_max : float or Quantity
            Include all baselines smaller than this number. Units of m.

        Returns
        -------
        array :
            1D array of regularly spaced un.
        """
        edges = self.xgrid_edges
        return (edges[:-1] + edges[1:]) / 2

    def ugrid_edges(self, frequency: tp.Frequency) -> np.ndarray:
        """Get a uv grid out to the maximum used baseline smaller than given bl_max.

        This is simply the :attribute:`xgrid_edges` converted to units of wavelengths
        at the given frequency.

        The resulting array represents the *edges* of the grid (so the number of cells
        is one fewer than this).

        Parameters
        ----------
        bl_max : float or Quantity
            Include all baselines smaller than this number. Units of length.

        Returns
        -------
        array :
            1D array of regularly spaced uv. Unitless.
        """
        return (self.xgrid_edges * frequency / cnst.c).to("")

    def ugrid(self, frequency: tp.Frequency) -> np.ndarray:
        """Get a uv grid out to the maximum used baseline smaller than given bl_max.

        This is simply the :attribute:`xgrid` converted to units of wavelengths
        at the given frequency.

        The resulting array represents the *centers* of the grid cells.

        Parameters
        ----------
        bl_max : float or Quantity
            Include all baselines smaller than this number. Units of length.

        Returns
        -------
        array :
            1D array of regularly spaced uv. Unitless.
        """
        return (self.xgrid * frequency / cnst.c).to("")

    def grid_baselines(
        self,
        coherent: bool,
        frequency: tp.Frequency | None = None,
        integration_time: tp.Time = 60.0 * un.s,
        observation_duration: tp.Time | None = None,
        phase_center_dec: tp.Angle | None = None,
        max_chunk_mem_gb: float = 1.0,
    ) -> np.ndarray:
        """
        Grid baselines onto a pre-determined uvgrid, accounting for earth rotation.

        Parameters
        ----------
        coherent
            Whether to coherently sum baselines within each uv cell, or
            incoherently sum them.
        frequency
            The frequency at which to grid the baselines. If not given, grids the
            baselines in bins of distance in meters.
        integration_time : float or Quantity, optional
            The amount of time integrated into a snapshot visibility.
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

        Notes
        -----
        The grid can either be in units of distance (meters) or in units of wavelengths,
        depending on the value of `frequency`. If `frequency` is None, the grid is in
        meters, which will yield the _same_ grid for all frequencies. If `frequency` is
        given, the grid is in units of wavelengths, and will be different for different
        frequencies (i.e. the grid will be constant in units of wavelengths, while the
        baselines themselves will change in units of wavelengths).

        For calculating the power spectrum sensitivity in 21cmSense, it is sufficient
        to use the same grid in units of meters for all frequencies. In general this
        should only cause very slight differences in the results, and it has the
        advantage of being much faster to compute over many frequencies/redshifts.

        If the :class:`Observatory` object is being used to generate noise realizations
        for other purposes, it may be important to use a grid in units of wavelengths.

        See Also
        --------
        grid_baselines_coherent :
            Coherent sum over baseline groups of the output of this method.
        grid_baseline_incoherent :
            Incoherent sum over baseline groups of the output of this method.
        """
        if observation_duration is None:
            # by default, assume a snapshot integration
            observation_duration = integration_time

        assert un.get_physical_type(integration_time) == "time"
        assert un.get_physical_type(observation_duration) == "time"

        time_offsets = self.time_offsets_from_obs_int_time(integration_time, observation_duration)

        if frequency is None:
            grid_edges = self.xgrid_edges
            frequencies = None
        else:
            grid_edges = self.ugrid_edges(frequency)
            frequencies = [frequency]

        du = grid_edges[1] - grid_edges[0]
        return ut.grid_baselines(
            coherent=coherent,
            baselines=self.redundant_baseline_vectors,
            weights=self.redundant_baseline_weights,
            time_offsets=time_offsets,
            frequencies=frequencies,
            ugrid_edges=grid_edges,
            vgrid_edges=grid_edges[grid_edges + du > 0],
            phase_center_dec=phase_center_dec,
            telescope_latitude=self.latitude,
            world=self.world,
            max_chunk_mem_gb=max_chunk_mem_gb,
        )[0]
