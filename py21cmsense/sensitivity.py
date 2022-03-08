"""
Classes fro which sensitivities can be obtained.

This module modularizes the previous version's `calc_sense.py`, and enables
multiple sensitivity kinds to be defined. By default, a PowerSpectrum sensitivity class
is provided, which offers the same results as previous versions.
"""
from __future__ import annotations

import attr
import h5py
import logging
import numpy as np
import os
import pickle
import tqdm
from astropy import units as un
from astropy.cosmology.units import littleh, with_H0
from astropy.io.misc import yaml
from attr import validators as vld
from cached_property import cached_property
from collections.abc import Mapping
from methodtools import lru_cache
from os import path
from pathlib import Path
from scipy import interpolate

from . import _utils as ut
from . import config
from . import conversions as conv
from . import observation as obs
from . import types as tp


def _kconverter(val, allow_unitless=False):
    if hasattr(val, "unit"):
        return val.to(littleh / un.Mpc, with_H0(config.COSMO.H0))
    if not allow_unitless:
        raise ValueError("no units supplied!")
    # Assume it has the 1/Mpc units (default from 21cmFAST)
    return (val / un.Mpc).to(littleh / un.Mpc, with_H0(config.COSMO.H0))


# Get default k, power
_K21_DEFAULT, _D21_DEFAULT = np.genfromtxt(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data/ps_no_halos_nf0.521457_z9.50_useTs0_zetaX-1.0e+00_200_400Mpc_v2",
    )
).T[:2]

_K21_DEFAULT = _kconverter(_K21_DEFAULT, allow_unitless=True)
_D21_DEFAULT <<= un.mK**2

logger = logging.getLogger(__name__)


@attr.s(kw_only=True)
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

    observation: obs.Observation = attr.ib(validator=vld.instance_of(obs.Observation))
    no_ns_baselines: bool = attr.ib(default=False, converter=bool)

    @staticmethod
    def _load_yaml(yaml_file):
        if isinstance(yaml_file, str):
            with open(yaml_file) as fl:
                data = yaml.load(fl)
        elif isinstance(yaml_file, Mapping):
            data = yaml_file
        else:
            raise ValueError(
                "yaml_file must be a string filepath or a raw dict from such a file."
            )
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
        elif obsfile.endswith(".pkl"):
            with open(obsfile, "rb") as fl:
                observation = pickle.load(fl)
        else:
            raise ValueError(
                "observation must be a filename with extension .yml or .pkl"
            )

        return klass(observation=observation, **data)


@attr.s(kw_only=True)
class PowerSpectrum(Sensitivity):
    """
    A Power Spectrum sensitivity calculator.

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
    """

    horizon_buffer: tp.Wavenumber = attr.ib(
        default=0.1 * littleh / un.Mpc,
        validator=(
            tp.vld_unit(littleh / un.Mpc, with_H0(config.COSMO.H0)),
            ut.nonnegative,
        ),
        converter=_kconverter,
    )
    foreground_model: str = attr.ib(
        default="moderate", validator=vld.in_(["moderate", "optimistic"])
    )
    k_21: tp.Wavenumber = attr.ib(
        _K21_DEFAULT,
        validator=tp.vld_unit(littleh / un.Mpc, with_H0(config.COSMO.H0)),
        converter=_kconverter,
    )
    delta_21: tp.Delta = attr.ib(_D21_DEFAULT, validator=(tp.vld_unit(un.mK**2)))

    @classmethod
    def from_yaml(cls, yaml_file) -> Sensitivity:
        """
        Construct a PowerSpectrum sensitivity from yaml.

        YAML spec has p21 as a file which obtains k_21 and delta_21.
        It is assumed that k in the file is in units of 1/Mpc (true for 21cmFAST).
        """
        data = cls._load_yaml(yaml_file)

        p21 = data.pop("p21", None)
        if p21 is not None:
            data["k_21"] = p21[:, 0] << 1 / un.Mpc
            data["delta_21"] = p21[:, 1] << un.mK**2

        if isinstance(yaml_file, str):
            obsfile = path.join(path.dirname(yaml_file), data.pop("observation"))
        else:
            obsfile = data.pop("observation")

        data["observation"] = obsfile

        return super().from_yaml(data)

    @k_21.validator
    def _p21k_validator(self, att, val):
        assert val.ndim == 1

    @delta_21.validator
    def _delta21_validator(self, att, val):
        assert val.ndim == 1
        assert val.shape == self.k_21.shape

    @cached_property
    def p21(self):
        """An interpolation function defining the cosmological power spectrum."""
        fnc = interpolate.interp1d(
            self.k_21.to_value(littleh / un.Mpc),
            self.delta_21.to_value(un.mK**2),
            kind="linear",
        )
        return lambda k: fnc(k) * un.mK**2

    @cached_property
    def k_min(self) -> tp.Wavenumber:
        """Minimum k value to use in estimates."""
        return self.k_21.min()

    @cached_property
    def k_max(self) -> tp.Wavenumber:
        """Maximum k value to use in estimates."""
        return self.k_21.max()

    @cached_property
    def k1d(self) -> tp.Wavenumber:
        """1D array of wavenumbers for which sensitivities will be generated."""
        delta = conv.dk_deta(self.observation.redshift) / self.observation.bandwidth
        if self.k_max.value < delta.value * self.observation.n_channels:
            logger.warning(
                "The maximum k value is being restricted by the theoretical signal "
                "model. Losing ~"
                f"{int((delta.value * self.observation.n_channels - self.k_max.value)/delta.value)}"
                " bins."
            )
        if self.k_min > delta:
            logger.warning(
                "The minimum k value is being restricted by the theoretical signal "
                f"model. Losing ~{int((self.k_min - delta)/delta)} bins between {delta}"
                f" and {self.k_min}."
            )
            mn = self.k_min
        else:
            mn = delta
        assert delta.unit == self.k_max.unit
        return np.arange(mn.value, self.k_max.value, delta.value) * delta.unit

    @cached_property
    def X2Y(self) -> un.Quantity[un.Mpc**3 / littleh**3 / un.steradian / un.GHz]:
        """Cosmological scaling factor X^2*Y (eg. Parsons 2012)."""
        return conv.X2Y(self.observation.redshift)

    @cached_property
    def uv_coverage(self) -> np.ndarray:
        """The UV-coverage of the array, with unused/redundant baselines set to zero."""
        grid = self.observation.uv_coverage.copy()
        size = grid.shape[0]

        # Cut unnecessary data out of uv coverage: auto-correlations & half of uv
        # plane (which is not statistically independent for real sky)
        grid[size // 2, size // 2] = 0.0
        grid[:, : size // 2] = 0.0
        grid[size // 2 :, size // 2] = 0.0

        if self.no_ns_baselines:
            grid[:, size // 2] = 0.0

        return grid

    def power_normalisation(self, k: tp.Wavenumber) -> float:
        """Normalisation constant for power spectrum."""
        assert k.unit.is_equivalent(littleh / un.Mpc)
        return (
            self.X2Y
            * self.observation.observatory.beam.b_eff
            * self.observation.bandwidth
            * k**3
            / (2 * np.pi**2)
        ).to_value("")

    def thermal_noise(
        self, k_par: tp.Wavenumber, k_perp: tp.Wavenumber, trms: tp.Temperature
    ) -> tp.Delta:
        """Thermal noise contribution at particular k mode."""
        k = np.sqrt(k_par**2 + k_perp**2)
        scalar = self.power_normalisation(k)
        return scalar * trms.to("mK") ** 2

    def sample_noise(self, k_par: tp.Wavenumber, k_perp: tp.Wavenumber) -> tp.Delta:
        """Sample variance contribution at a particular k mode."""
        k = np.sqrt(k_par**2 + k_perp**2)
        vals = np.full(k.size, np.inf) * un.mK**2
        good_ks = np.logical_and(k >= self.k_min, k <= self.k_max)
        vals[good_ks] = self.p21(k[good_ks])
        return vals

    @cached_property
    def _nsamples_2d(
        self,
    ) -> dict[str, dict[tp.Wavenumber, un.Quantity[1 / un.mK**4]]]:
        """Mid-way product specifying thermal and sample variance over the 2D grid."""
        # set up blank arrays/dictionaries
        sense = {"sample": {}, "thermal": {}, "both": {}}

        # loop over uv_coverage to calculate k_pr
        nonzero = np.where(self.uv_coverage > 0)
        for iu, iv in tqdm.tqdm(
            zip(nonzero[1], nonzero[0]),
            desc="calculating 2D sensitivity",
            unit="uv-bins",
            disable=not config.PROGRESS,
            total=len(nonzero[1]),
        ):
            u, v = self.observation.ugrid[iu], self.observation.ugrid[iv]
            trms = self.observation.Trms[iv, iu]

            if np.isinf(trms):
                continue

            umag = np.sqrt(u**2 + v**2)
            k_perp = umag * conv.dk_du(self.observation.redshift)

            hor = self.horizon_limit(umag)

            if k_perp not in sense["thermal"]:
                sense["thermal"][k_perp] = (
                    np.zeros(len(self.observation.kparallel)) / un.mK**4
                )
                sense["sample"][k_perp] = (
                    np.zeros(len(self.observation.kparallel)) / un.mK**4
                )
                sense["both"][k_perp] = (
                    np.zeros(len(self.observation.kparallel)) / un.mK**4
                )

            # Exclude parallel modes dominated by foregrounds
            kpars = self.observation.kparallel[self.observation.kparallel >= hor]

            if not len(kpars):
                continue

            start = np.where(self.observation.kparallel >= hor)[0][0]
            n_inds = (self.observation.kparallel.size - 1) // 2 + 1
            inds = np.arange(start=start, stop=n_inds)

            thermal = self.thermal_noise(kpars, k_perp, trms)
            sample = self.sample_noise(kpars, k_perp)

            t = 1.0 / thermal**2
            s = 1.0 / sample**2
            ts = 1.0 / (thermal + sample) ** 2
            sense["thermal"][k_perp][inds] += t
            sense["thermal"][k_perp][-inds] += t
            sense["sample"][k_perp][inds] += s
            sense["sample"][k_perp][-inds] += s
            sense["both"][k_perp][inds] += ts
            sense["both"][k_perp][-inds] += ts

        return sense

    @lru_cache()
    def calculate_sensitivity_2d(
        self, thermal: bool = True, sample: bool = True
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
        for k_perp in sense.keys():
            mask = sense[k_perp] > 0
            final_sense[k_perp] = np.inf * np.ones(len(mask)) * un.mK**2
            final_sense[k_perp][mask] = sense[k_perp][mask] ** -0.5 / np.sqrt(
                self.observation.n_lst_bins
            )

        return final_sense

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
            conv.dk_deta(self.observation.redshift) * umag / self.observation.frequency
        )
        # calculate horizon limit for baseline of length umag
        if self.foreground_model in ["moderate", "pessimistic"]:
            return horizon + self.horizon_buffer
        elif self.foreground_model in ["optimistic"]:
            return horizon * np.sin(self.observation.observatory.beam.first_null / 2)

    def _average_sense_to_1d(self, sense: dict[tp.Wavenumber, tp.Delta]) -> tp.Delta:
        """Bin 2D sensitivity down to 1D."""
        sense1d_inv = np.zeros(len(self.k1d)) / un.mK**4

        for k_perp in tqdm.tqdm(
            sense.keys(),
            desc="averaging to 1D",
            unit="kperp-bins",
            disable=not config.PROGRESS,
        ):
            k = np.sqrt(self.observation.kparallel**2 + k_perp**2)
            good_ks = np.logical_and(self.k_min <= k, k <= self.k_max)
            sense1d_inv[ut.find_nearest(self.k1d, k[good_ks])] += (
                1.0 / sense[k_perp][good_ks] ** 2
            )

        # invert errors and take square root again for final answer
        sense1d = np.ones(sense1d_inv.shape) * un.mK**2 * np.inf
        mask = sense1d_inv > 0
        sense1d[mask] = 1 / np.sqrt(sense1d_inv[mask])
        return sense1d

    @lru_cache()
    def calculate_sensitivity_1d(
        self, thermal: bool = True, sample: bool = True
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
        sense = self.calculate_sensitivity_2d(thermal=thermal, sample=sample)
        return self._average_sense_to_1d(sense)

    @property
    def delta_squared(self) -> tp.Delta:
        """The fiducial 21cm power spectrum evaluated at :attr:`k1d`."""
        return self.p21(self.k1d)

    @lru_cache()
    def calculate_significance(
        self, thermal: bool = True, sample: bool = True
    ) -> float:
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
        mask = np.logical_and(self.k1d >= self.k_min, self.k1d <= self.k_max)
        sense1d = self.calculate_sensitivity_1d(thermal=thermal, sample=sample)

        snr = self.delta_squared[mask] / sense1d[mask]
        return np.sqrt(float(np.dot(snr, snr.T)))

    def plot_sense_2d(self, sense2d: dict[tp.Wavenumber, tp.Delta]):
        """Create a colormap plot of the sensitivity un UV bins."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:  # pragma: no cover
            raise ImportError("matplotlib is required to make plots...")

        keys = sorted(sense2d.keys())
        x = np.array([v.value for v in keys])
        x = (
            np.repeat(x, len(self.observation.kparallel))
            .reshape((len(x), len(self.observation.kparallel)))
            .T
        )
        y = np.fft.fftshift(
            np.repeat(self.observation.kparallel.value, x.shape[1]).reshape(
                (len(self.observation.kparallel), x.shape[1])
            )
        )
        z = np.array([np.fft.fftshift(sense2d[key]) for key in keys]).T

        plt.pcolormesh(x, y, np.log10(z))
        cbar = plt.colorbar()
        cbar.set_label(r"$\log_{10} \delta \Delta^2$ [mK^2]", fontsize=14)
        plt.xlabel(r"$k_\perp$ [h/Mpc]", fontsize=14)
        plt.ylabel(r"$k_{||}$ [h/Mpc]", fontsize=14)

    def write(
        self,
        filename: str | Path,
        thermal: bool = True,
        sample: bool = True,
        prefix: str = None,
    ) -> Path:
        """Save sensitivity results to HDF5 file.

        Returns
        -------
        filename
            The path to the file that is written.
        """
        out = self._get_all_sensitivity_combos(thermal, sample)
        prefix = prefix + "_" if prefix else ""
        if filename is None:
            filename = Path(
                f"{prefix}{self.foreground_model}_{self.observation.frequency:.3f}.h5"
            )
        else:
            filename = Path(filename)

        logger.info(f"Writing sensitivies to '{filename}'")
        with h5py.File(filename, "w") as fl:

            # TODO: We should be careful to try and write everything into this file
            # i.e. all the parameters etc.

            for k, v in out.items():
                fl[k] = v
                fl[k.replace("noise", "snr")] = self.delta_squared / v

            fl["k"] = self.k1d.to("1/Mpc", with_H0(config.COSMO.H0)).value
            fl["delta_squared"] = self.delta_squared

            fl.attrs["k_min"] = self.k_min
            fl.attrs["k_max"] = self.k_max
            fl.attrs["total_snr"] = self.calculate_significance()
            fl.attrs["foreground_model"] = self.foreground_model
            fl.attrs["horizon_buffer"] = self.horizon_buffer
            fl.attrs["k_unit"] = "1/Mpc"

        return filename

    def plot_sense_1d(self, sample: bool = True, thermal: bool = True):
        """Create a plot of the sensitivity in 1D k-bins."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:  # pragma: no cover
            raise ImportError("matplotlib is required to make plots...")

        out = self._get_all_sensitivity_combos(thermal, sample)
        for key, value in out.items():
            plt.plot(self.k1d, value, label=key)
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel("k [1/Mpc]")
            plt.ylabel(r"$\Delta^2_N \  [{\rm mK}^2/{\rm Mpc}^3$")
            plt.legend()
            plt.title(f"z={conv.f2z(self.observation.frequency):.2f}")

        return plt.gcf()

    def _get_all_sensitivity_combos(
        self, thermal: bool, sample: bool
    ) -> dict[str, tp.Delta]:
        result = {}
        if thermal:
            result["thermal_noise"] = self.calculate_sensitivity_1d(sample=False)
        if sample:
            result["sample_noise"] = self.calculate_sensitivity_1d(
                thermal=False, sample=True
            )

        if thermal and sample:
            result["sample+thermal_noise"] = self.calculate_sensitivity_1d(
                thermal=True, sample=True
            )

        return result
