"""
Classes from which sensitivities can be obtained.

This module modularizes the previous version's `calc_sense.py`, and enables
multiple sensitivity kinds to be defined. By default, a :class:`PowerSpectrum`
sensitivity class is provided, which offers the same results as previous versions.
In the future, we may provide things like ``ImagingSensitivity`` or
``WaveletSensitivity`` for example.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable, Mapping
from functools import cached_property
from os import path
from pathlib import Path

import attrs
import h5py
import hickle
import numpy as np
import tqdm
from astropy import units as un
from astropy.cosmology import LambdaCDM
from astropy.cosmology.units import littleh, with_H0
from astropy.io.misc import yaml
from attr import validators as vld
from hickleable import hickleable
from methodtools import lru_cache

from . import _utils as ut
from . import config
from . import conversions as conv
from . import observation as obs
from . import units as tp
from .theory import _ALL_THEORY_POWER_SPECTRA, EOS2021, TheoryModel

logger = logging.getLogger(__name__)


@hickleable(evaluate_cached_properties=True)
@attrs.define(kw_only=True, slots=False)
class Sensitivity:
    """
    Base class for sensitivity calculations.

    Parameters
    ----------
    observation : :class:`~py21cmsense.Observation` instance
        An object defining the observatory and observation used to derive sensitivity.
    no_ns_baselines : bool, optional
        Remove pure north/south baselines (u=0) from the sensitivity calculation.
        These baselines can potentially have higher systematics, so excluding them
        represents a conservative choice.
    """

    observation: obs.Observation = attrs.field(validator=vld.instance_of(obs.Observation))
    no_ns_baselines: bool = attrs.field(default=False, converter=bool)

    @staticmethod
    def _load_yaml(yaml_file):
        if isinstance(yaml_file, str):
            with open(yaml_file) as fl:
                data = yaml.load(fl)
        elif isinstance(yaml_file, Mapping):
            data = yaml_file
        else:
            raise ValueError("yaml_file must be a string filepath or a raw dict from such a file.")
        return data

    @classmethod
    def from_yaml(cls, yaml_file) -> Sensitivity:
        """Construct a :class:`Sensitivity` object from a YAML configuration."""
        data = cls._load_yaml(yaml_file)

        klass = data.pop("class", cls)

        if isinstance(yaml_file, str) and not path.isabs(data["observation"]):
            obsfile = path.join(path.dirname(yaml_file), data.pop("observation"))
        else:
            obsfile = data.pop("observation")

        if obsfile.endswith(".yml"):
            observation = obs.Observation.from_yaml(obsfile)
        elif h5py.is_hdf5(obsfile):
            observation = hickle.load(obsfile)
        else:
            raise ValueError("observation must be a filename with extension .yml or .h5")

        return klass(observation=observation, **data)

    def clone(self, **kwargs):
        """Clone the object with new parameters."""
        return attrs.evolve(self, **kwargs)

    def at_frequency(self, frequency: un.Quantity[un.MHz]) -> Sensitivity:
        """Return a new object at a new frequency."""
        return self.clone(observation=self.observation.clone(frequency=frequency))

    @property
    def cosmo(self) -> LambdaCDM:
        """The cosmology to use in the sensitivity calculations."""
        return self.observation.cosmo

    @property
    def frequency(self) -> un.Quantity[un.MHz]:
        """The frequency of the observation."""
        return self.observation.frequency


@attrs.define(kw_only=True, slots=False)
class PowerSpectrum(Sensitivity):
    """
    A Power Spectrum sensitivity calculator.

    Note that the sensitivity calculation contains two major parts: thermal variance
    and sample variance (aka cosmic variance). The latter requires a model of the power
    spetrum itself, which you should provide via ``k_21`` and ``delta_21``.
    Remember that the power spectrum is redshift dependent, and so should be supplied
    differently at each frequency being calculated.

    Parameters
    ----------
    horizon_buffer : float or Quantity
        A buffer to add to the horizon line in order to excise foreground-contaminated modes.
    foreground_model : str, {moderate, optimistic}
        Which approach to take for foreground excision. Moderate uses a defined horizon buffer,
        while optimistic excludes all k modes inside the primary field of view.
    k_21 : array or Quantity, optional
        An array of wavenumbers used to define a cosmological power spectrum, in order to get
        sample variance. If not a Quantity, will assume k has units of 1/Mpc, though it will
        convert these units to h/Mpc throughout the class. Default is to use built-in
        data file from 21cmFAST.
    delta_21 : array or Quantity, optional
        An array of Delta^2 power spectrum values used for sample variance.
        If not a Quantity, will assume units of mK^2.
    systematics_mask : callable
        A function that takes a single kperp and an array of kpar, and returns a boolean
        array specifying which of the k's are useable after accounting for systematics.
        that is, it returns False for k's affected by systematics.
    """

    horizon_buffer: tp.Wavenumber = attrs.field(default=0.1 * littleh / un.Mpc)
    foreground_model: str = attrs.field(
        default="moderate", validator=vld.in_(["moderate", "optimistic", "foreground_free"])
    )
    theory_model: TheoryModel = attrs.field()
    systematics_mask: Callable | None = attrs.field(default=None)

    @horizon_buffer.validator
    def _horizon_buffer_validator(self, att, val):
        tp.vld_unit(littleh / un.Mpc, with_H0(self.cosmo.H0))(self, att, val)
        ut.nonnegative(self, att, val)

    @classmethod
    def from_yaml(cls, yaml_file) -> Sensitivity:
        """
        Construct a PowerSpectrum sensitivity from yaml.

        YAML spec has p21 as a file which obtains k_21 and delta_21.
        It is assumed that k in the file is in units of 1/Mpc (true for 21cmFAST).
        """
        data = cls._load_yaml(yaml_file)

        if "plugins" in data:
            if not isinstance(data["plugins"], list):
                raise ValueError(
                    "plugins in YAML file must be a list of modules."
                )  # pragma: no cover

            for mdl in data.pop("plugins"):
                try:
                    importlib.import_module(mdl)
                except Exception as e:
                    raise ImportError(f"Could not import {mdl}") from e

        if "theory_model" in data:
            data["theory_model"] = _ALL_THEORY_POWER_SPECTRA[data["theory_model"]]()

        if isinstance(yaml_file, str):
            obsfile = path.join(path.dirname(yaml_file), data.pop("observation"))
        else:
            obsfile = data.pop("observation")

        data["observation"] = obsfile

        return super().from_yaml(data)

    @theory_model.default
    def _theory_model_default(self):
        return EOS2021()

    @theory_model.validator
    def _theory_model_validator(self, att, val):
        if not isinstance(val, TheoryModel):
            raise ValueError("The theory_model must be an instance of TheoryModel")

    @cached_property
    def k1d(self) -> tp.Wavenumber:
        """1D array of wavenumbers for which sensitivities will be generated."""
        delta = (
            conv.dk_deta(
                self.observation.redshift,
                self.cosmo,
                approximate=self.observation.use_approximate_cosmo,
            )
            / self.observation.bandwidth
        )
        dv = delta.value
        return np.arange(dv, dv * self.observation.n_channels, dv) * delta.unit

    @cached_property
    def X2Y(self) -> un.Quantity[un.Mpc**3 / littleh**3 / un.steradian / un.GHz]:
        """Cosmological scaling factor X^2*Y (eg. Parsons 2012)."""
        return conv.X2Y(
            self.observation.redshift,
            approximate=self.observation.use_approximate_cosmo,
        )

    @cached_property
    def uv_coverage(self) -> np.ndarray:
        """The UV-coverage of the array, with unused/redundant baselines set to zero."""
        grid = self.observation.uv_coverage.copy()

        if self.no_ns_baselines:
            grid[:, 0] = 0.0

        return grid

    @cached_property
    def kparallel(self) -> tp.Wavenumber:
        """The kparallel magnitudes.

        Only positive kparallel are returned, since negative kparallel have the same
        sensitivity and so are redundant for our calculations.
        """
        return self.observation.kparallel[self.observation.kparallel > 0]

    def power_normalisation(self, k: tp.Wavenumber) -> float:
        """Normalisation constant for power spectrum."""
        assert hasattr(k, "unit")
        assert k.unit.is_equivalent(littleh / un.Mpc)

        return (
            self.X2Y
            * self.observation.observatory.beam.b_eff(self.frequency)
            * self.observation.bandwidth
            * k**3
            / (2 * np.pi**2)
        ).to_value("")

    def thermal_noise(self, k: tp.Wavenumber, trms: tp.Temperature) -> tp.Delta:
        """Thermal noise contribution at particular k mode."""
        scalar = self.power_normalisation(k)
        return scalar * trms.to("mK") ** 2

    def sample_noise(self, k: tp.Wavenumber) -> tp.Delta:
        """Sample variance contribution at a particular k mode."""
        k = k.to_value(
            littleh / un.Mpc if self.theory_model.use_littleh else un.Mpc**-1,
            with_H0(self.cosmo.H0),
        )
        return self.theory_model.delta_squared(self.observation.redshift, k)

    @cached_property
    def _nsamples_2d(
        self,
    ) -> dict[str, dict[tp.Wavenumber, un.Quantity[1 / un.mK**4]]]:
        """Mid-way product specifying thermal and sample variance over the 2D grid."""
        # set up blank arrays/dictionaries
        sense = {"sample": {}, "thermal": {}, "both": {}}

        n2 = len(self.uv_coverage[:, 0]) // 2

        ugrid = self.observation.ugrid
        umag = np.sqrt(np.add.outer(ugrid**2, ugrid[n2:] ** 2))

        k_perps = umag.flatten() * conv.dk_du(
            self.observation.redshift,
            self.cosmo,
            approximate=self.observation.use_approximate_cosmo,
        )

        horizon_limits = self.horizon_limit(umag).flatten()

        mask = (self.kparallel.max() >= horizon_limits) & (
            ~np.isinf(self.observation.Trms.flatten())
        )

        k_perps = k_perps[mask]
        horizon_limits = horizon_limits[mask]
        trms_vals = self.observation.Trms.flatten()[mask]

        # Only take positive kparallel since negatives have the same
        # sensitivity and are dealt with when combining the info later.
        kpar = self.kparallel

        # This could turn into a very big array. Maybe there's a better
        # way to do this?
        k = np.sqrt(kpar[:, None] ** 2 + k_perps[None, :] ** 2)
        thermal = self.thermal_noise(k, trms_vals)
        sample = self.sample_noise(k)

        for k_perp, hor, tnoise, snoise in tqdm.tqdm(
            zip(k_perps, horizon_limits, thermal.T, sample.T, strict=False),
            desc="calculating 2D sensitivity",
            unit="uv-bins",
            disable=not config.PROGRESS,
            total=len(k_perps),
        ):
            # Exclude parallel modes dominated by foregrounds
            mask = kpar >= hor

            if k_perp not in sense["thermal"]:
                sense["thermal"][k_perp] = np.zeros(len(kpar)) / un.mK**4
                sense["sample"][k_perp] = np.zeros(len(kpar)) / un.mK**4
                sense["both"][k_perp] = np.zeros(len(kpar)) / un.mK**4

            # The following assumes that the power spectra are averaged with inverse
            # variance weighting. The scaling by 2 is due to the fact that we add
            # both positive and negative k_parallel contributions together, which are
            # independent measurements of the same underlying power spectrum mode.
            t = 1.0 / tnoise[mask] ** 2
            s = 1.0 / snoise[mask] ** 2
            ts = 1.0 / (tnoise[mask] + snoise[mask]) ** 2
            sense["thermal"][k_perp][mask] += 2 * t
            sense["sample"][k_perp][mask] += 2 * s
            sense["both"][k_perp][mask] += 2 * ts

        return sense

    @lru_cache()
    def calculate_sensitivity_2d(
        self,
        thermal: bool = True,
        sample: bool = True,
        compute_all: bool = False,
    ) -> dict[tp.Wavenumber, tp.Delta]:
        """
        Calculate power spectrum sensitivity for a grid of cylindrical k modes.

        Parameters
        ----------
        thermal : bool, optional
            Whether to calculate thermal contribution to the sensitivity
        sample : bool, optional
            Whether to calculate sample variance contribution to sensitivity.

        Returns
        -------
        dict :
            Keys are cylindrical kperp values and values are arrays aligned with
            `observation.kparallel`, defining uncertainty in mK^2.
        """
        if thermal and sample:
            logger.info("Getting Combined Variance")
            sense = self._nsamples_2d["both"]
        elif thermal:
            logger.info("Getting Thermal Variance")
            sense = self._nsamples_2d["thermal"]
        elif sample:
            logger.info("Getting Sample Variance")
            sense = self._nsamples_2d["sample"]
        else:
            raise ValueError("Either thermal or sample must be True")

        # errors were added in inverse quadrature, now need to invert and take
        # square root to have error bars; also divide errors by number of indep. fields
        final_sense = {}
        if compute_all:
            final_sense["thermal"] = {}
            final_sense["sample"] = {}
            final_sense["both"] = {}

        for k_perp in sense:
            mask = sense[k_perp] > 0
            if self.systematics_mask is not None:
                mask &= self.systematics_mask(k_perp, self.kparallel)

            if not np.any(mask):
                continue

            if compute_all:
                final_sense["thermal"][k_perp] = np.inf * np.ones(len(mask)) * un.mK**2
                final_sense["sample"][k_perp] = np.inf * np.ones(len(mask)) * un.mK**2
                final_sense["both"][k_perp] = np.inf * np.ones(len(mask)) * un.mK**2
            else:
                final_sense[k_perp] = np.inf * np.ones(len(mask)) * un.mK**2

            thermal_std = 0
            if thermal or compute_all:
                total_std = thermal_std = 1 / np.sqrt(
                    self._nsamples_2d["thermal"][k_perp][mask] * self.observation.n_lst_bins
                )

            sample_std = 0
            if sample or compute_all:
                total_std = sample_std = 1 / np.sqrt(
                    self._nsamples_2d["sample"][k_perp][mask]
                    * (self.observation.time_per_day / self.observation.lst_bin_size).to("")
                )
            if (thermal and sample) or compute_all:
                total_std = thermal_std + sample_std

            if compute_all:
                final_sense["thermal"][k_perp][mask] = thermal_std
                final_sense["sample"][k_perp][mask] = sample_std
                final_sense["both"][k_perp][mask] = total_std
            else:
                final_sense[k_perp][mask] = total_std

        return final_sense

    def calculate_sensitivity_2d_grid(
        self,
        kperp_edges: tp.Wavenumber,
        kpar_edges: tp.Wavenumber,
        thermal: bool = True,
        sample: bool = True,
        compute_all: bool = False,
    ) -> tp.Delta:
        """Calculate the 2D cylindrical sensitivity on a grid of kperp/kpar.

        Parameters
        ----------
        kperp_edges
            The edges of the bins in kperp.
        kpar_edges
            The edges of the bins in kpar.
        """
        if compute_all:
            sense2d_inv = {
                "thermal": np.zeros((len(kperp_edges) - 1, len(kpar_edges) - 1)) / un.mK**4,
                "sample": np.zeros((len(kperp_edges) - 1, len(kpar_edges) - 1)) / un.mK**4,
                "both": np.zeros((len(kperp_edges) - 1, len(kpar_edges) - 1)) / un.mK**4,
            }
        else:
            sense2d_inv = np.zeros((len(kperp_edges) - 1, len(kpar_edges) - 1)) << (1 / un.mK**4)

        sense = self.calculate_sensitivity_2d(compute_all=True)

        assert np.all(np.diff(kperp_edges) > 0)
        assert np.all(np.diff(kpar_edges) > 0)

        for k_perp in tqdm.tqdm(
            sense["thermal"].keys(),
            desc="averaging to 2D grid",
            unit="kperp-bins",
            disable=not config.PROGRESS,
        ):
            if k_perp < kperp_edges[0] or k_perp >= kperp_edges[-1]:
                continue

            # Get the kperp bin it's in.
            kperp_indx = np.where(k_perp >= kperp_edges)[0][-1]

            kpar_indx = np.digitize(self.kparallel, kpar_edges) - 1
            good_ks = kpar_indx >= 0
            good_ks &= kpar_indx < len(kpar_edges) - 1

            if compute_all:
                for key in ["thermal", "sample", "both"]:
                    sense2d_inv[key][kperp_indx][kpar_indx[good_ks]] += (
                        1.0 / sense[key][k_perp][good_ks] ** 2
                    )
            else:
                key = "both" if (thermal and sample) else "thermal" if thermal else "sample"
                sense2d_inv[kperp_indx][kpar_indx[good_ks]] += (
                    1.0 / sense[key][k_perp][good_ks] ** 2
                )

        # invert errors and take square root again for final answer
        if compute_all:
            final_sense2d = {}
            for key in ["thermal", "sample", "both"]:
                sense2d = np.ones(sense2d_inv[key].shape) * un.mK**2 * np.inf
                mask = sense2d_inv[key] > 0
                sense2d[mask] = 1 / np.sqrt(sense2d_inv[key][mask])
                final_sense2d[key] = sense2d
            return final_sense2d
        else:
            sense2d = np.ones(sense2d_inv.shape) * un.mK**2 * np.inf
            mask = sense2d_inv > 0
            sense2d[mask] = 1 / np.sqrt(sense2d_inv[mask])
            return sense2d

    def horizon_limit(self, umag: float) -> tp.Wavenumber:
        """
        Calculate a horizon limit, with included buffer, if appropriate.

        Parameters
        ----------
        umag : float
            Baseline length (in wavelengths) at which to compute the horizon limit.

        Returns
        -------
        float :
            Horizon limit, in h/Mpc.
        """
        horizon = (
            conv.dk_deta(
                self.observation.redshift,
                self.cosmo,
                approximate=self.observation.use_approximate_cosmo,
            )
            * umag
            / self.observation.frequency
        )
        # calculate horizon limit for baseline of length umag
        if self.foreground_model in ["moderate", "pessimistic"]:
            return horizon + self.horizon_buffer
        elif self.foreground_model in ["optimistic"]:
            beam = self.observation.observatory.beam
            return horizon * np.sin(beam.first_null(self.frequency) / 2)
        elif self.foreground_model in ["foreground_free"]:
            return 0

    def _average_sense_to_1d(
        self,
        sense: dict[tp.Wavenumber, tp.Delta] | dict[str, dict[tp.Wavenumber, tp.Delta]],
        k1d: tp.Wavenumber | None = None,
        compute_all: bool = False,
    ) -> tp.Delta:
        """Bin 2D sensitivity down to 1D."""
        if k1d is None:
            k1d = self.k1d

        if compute_all:
            sense1d_inv = {
                "thermal": np.zeros(len(k1d)) / un.mK**4,
                "sample": np.zeros(len(k1d)) / un.mK**4,
                "both": np.zeros(len(k1d)) / un.mK**4,
            }
            k_perps = list(sense["thermal"].keys())
        else:
            sense1d_inv = np.zeros(len(k1d)) / un.mK**4
            k_perps = list(sense.keys())

        for k_perp in tqdm.tqdm(
            k_perps,
            desc="averaging to 1D",
            unit="kperp-bins",
            disable=not config.PROGRESS,
        ):
            k = np.sqrt(self.kparallel**2 + k_perp**2)

            good_ks = k >= k1d.min()
            good_ks &= k < k1d.max()

            for cnt, kbin in enumerate(ut.find_nearest(k1d, k[good_ks])):
                if compute_all:
                    for key in ["thermal", "sample", "both"]:
                        sense1d_inv[key][kbin] += 1.0 / sense[key][k_perp][good_ks][cnt] ** 2
                else:
                    sense1d_inv[kbin] += 1.0 / sense[k_perp][good_ks][cnt] ** 2

        # invert errors and take square root again for final answer
        if compute_all:
            sense1d = {}
            for key in ["thermal", "sample", "both"]:
                sense1d[key] = np.ones(sense1d_inv[key].shape) * un.mK**2 * np.inf
                mask = sense1d_inv[key] > 0
                sense1d[key][mask] = 1 / np.sqrt(sense1d_inv[key][mask])
        else:
            sense1d = np.ones(sense1d_inv.shape) * un.mK**2 * np.inf
            mask = sense1d_inv > 0
            sense1d[mask] = 1 / np.sqrt(sense1d_inv[mask])
        return sense1d

    @lru_cache()
    def calculate_sensitivity_1d(
        self,
        thermal: bool = True,
        sample: bool = True,
        compute_all: bool = False,
    ) -> tp.Delta:
        """Calculate a 1D sensitivity curve.

        Parameters
        ----------
        thermal
            Whether to calculate thermal contribution to the sensitivity
        sample
            Whether to calculate sample variance contribution to sensitivity.

        Returns
        -------
        array :
            1D array with units mK^2... the variance of spherical k modes.
        """
        sense = self.calculate_sensitivity_2d(
            thermal=thermal, sample=sample, compute_all=compute_all
        )
        return self._average_sense_to_1d(sense, compute_all=compute_all)

    def calculate_sensitivity_1d_binned(
        self, k: tp.Wavenumber, compute_all: bool = False, **kwargs
    ):
        """Calculate the 1D sensitivity at arbitrary k-bins."""
        sense2d = self.calculate_sensitivity_2d(**kwargs, compute_all=compute_all)
        return self._average_sense_to_1d(sense2d, k1d=k, compute_all=compute_all)

    @property
    def delta_squared(self) -> tp.Delta:
        """The fiducial 21cm power spectrum evaluated at :attr:`k1d`."""
        k = self.k1d.to_value(
            littleh / un.Mpc if self.theory_model.use_littleh else un.Mpc**-1,
            with_H0(self.cosmo.H0),
        )
        return self.theory_model.delta_squared(self.observation.redshift, k)

    @lru_cache()
    def calculate_significance(self, thermal: bool = True, sample: bool = True) -> float:
        """
        Calculate significance of a detection of the default cosmological power spectrum.

        Parameters
        ----------
        thermal : bool, optional
            Whether to calculate thermal contribution to the sensitivity
        sample : bool, optional
            Whether to calculate sample variance contribution to sensitivity.

        Returns
        -------
        float :
            Significance of detection (in units of sigma).
        """
        sense1d = self.calculate_sensitivity_1d(thermal=thermal, sample=sample)

        snr = self.delta_squared / sense1d
        return np.sqrt(float(np.dot(snr, snr.T)))

    def plot_sense_2d(self, sense2d: dict[tp.Wavenumber, tp.Delta]):
        # sourcery skip: raise-from-previous-error
        """Create a colormap plot of the sensitivity un UV bins."""
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:  # pragma: no cover
            raise ImportError("matplotlib is required to make plots...") from e

        keys = sorted(sense2d.keys())
        x = np.array([v.value for v in keys])
        x = np.repeat(x, len(self.kparallel)).reshape((len(x), len(self.kparallel))).T
        y = np.repeat(self.kparallel.value, x.shape[1]).reshape((len(self.kparallel), x.shape[1]))

        z = np.array([sense2d[key] for key in keys]).T

        plt.pcolormesh(x, y, np.log10(z))
        cbar = plt.colorbar()
        cbar.set_label(r"$\log_{10} \delta \Delta^2\ \  [{\rm mK}^2]$", fontsize=14)
        plt.xlabel(r"$k_\perp$ [h/Mpc]", fontsize=14)
        plt.ylabel(r"$k_{||}$ [h/Mpc]", fontsize=14)

    def write(
        self,
        filename: str | Path,
        thermal: bool = True,
        sample: bool = True,
        prefix: str | None = None,
        direc: str | Path = ".",
    ) -> Path:
        """Save sensitivity results to HDF5 file.

        Returns
        -------
        filename
            The path to the file that is written.
        """
        out = self.calculate_sensitivity_1d(compute_all=True)
        prefix = f"{prefix}_" if prefix else ""
        if filename is None:
            filename = Path(
                f"{prefix}{self.foreground_model}_{self.observation.frequency:.3f}.h5".replace(
                    " ", ""
                )
            )
        else:
            filename = Path(filename)

        if direc is not None:
            filename = Path(direc) / filename

        logger.info(f"Writing sensitivies to '{filename}'")
        with h5py.File(filename, "w") as fl:
            # TODO: We should be careful to try and write everything into this file
            # i.e. all the parameters etc.

            for k, v in out.items():
                if k == "both":
                    k = "sample+thermal"
                fl[k] = v
                fl[f"{k}_snr"] = self.delta_squared / v

            fl["k"] = self.k1d.to("1/Mpc", with_H0(self.cosmo.H0)).value
            fl["delta_squared"] = self.delta_squared

            fl.attrs["total_snr"] = self.calculate_significance()
            fl.attrs["foreground_model"] = self.foreground_model
            fl.attrs["horizon_buffer"] = self.horizon_buffer
            fl.attrs["k_unit"] = "1/Mpc"

        return filename

    def plot_sense_1d(self):
        """Create a plot of the sensitivity in 1D k-bins."""
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:  # pragma: no cover
            raise ImportError("matplotlib is required to make plots...") from e

        out = self.calculate_sensitivity_1d(compute_all=True)
        for key, value in out.items():
            plt.plot(self.k1d, value, label=key)

            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel("k [h/Mpc]")
            plt.ylabel(r"$\Delta^2_N \  [{\rm mK}^2]$")
            plt.legend()
            plt.title(f"z={conv.f2z(self.observation.frequency):.2f}")
        return plt.gcf()
