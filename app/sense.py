import marimo

__generated_with = "0.1.79"
app = marimo.App()


@app.cell
def __():
    import marimo as mo

    return (mo,)


@app.cell
def __():
    import attrs
    import numpy as np
    from astropy import units
    from astropy.cosmology.units import littleh
    from matplotlib import pyplot as plt
    from matplotlib.colors import LogNorm

    import py21cmsense as p21c
    from py21cmsense.theory import _ALL_THEORY_POWER_SPECTRA

    return LogNorm, attrs, littleh, np, p21c, plt, units


@app.cell
def __(mo, p21c):
    observatory_inputs = {
        "profile": mo.ui.dropdown(
            options=p21c.observatory.get_builtin_profiles(),
            value=p21c.observatory.get_builtin_profiles()[0],
            label="Telescope Profile:",
        ),
        "frequency": mo.ui.slider(
            start=50, stop=350, step=1, value=150, label="Frequency [MHz]"
        ),
    }
    observatory_form = (
        mo.md("\n\n".join("{%s}" % key for key in observatory_inputs))
        .batch(**observatory_inputs)
        .form()
    )
    return observatory_form, observatory_inputs


@app.cell
def __():
    return


@app.cell
def __(mo):
    observation_inputs = {
        "time_per_day": mo.ui.slider(1, 24, value=6, label="Hours observed per day"),
        # "lst_bin_size": mo.ui.slider(1, 24, value=),
        "integration_time": mo.ui.slider(
            1, 86400, value=10, label="Integration Time (s)"
        ),
        "n_channels": mo.ui.slider(10, 300, value=82, label=r"\# Channels"),
        "bandwidth": mo.ui.slider(4, 40, value=8.0, label="Bandwidth (MHz)"),
        "n_days": mo.ui.slider(1, 360, value=180, label="Number of days observed"),
        #'baseline_filters',
        "redundancy_tol": mo.ui.slider(
            1, 10, value=1, label=r"\# Decimal places to compare redundancy"
        ),
        "coherent": mo.ui.checkbox(label="Coherently Average Snapshots?"),
        "spectral_index": mo.ui.slider(
            1, 4, step=0.1, value=2.6, label="Spectral Index (negative) of sky"
        ),
        "tsky_amplitude": mo.ui.slider(
            50, 500, step=1, value=260, label="Sky reference temperature (K)"
        ),
        "tsky_ref_freq": mo.ui.slider(
            50, 1000, step=1, value=150, label="Sky reference frequency (MHz)"
        ),
    }

    observation_form = (
        mo.md("\n\n".join("{%s}" % key for key in observation_inputs))
        .batch(**observation_inputs)
        .form()
    )
    return observation_form, observation_inputs


@app.cell
def __(mo):
    sense_inputs = {
        "no_ns_baselines": mo.ui.checkbox(label="Remove North-South Baselines?"),
        "horizon_buffer": mo.ui.slider(0, 1, value=0.1, label="Horizon Buffer (h/Mpc)"),
        "foreground_model": mo.ui.radio(
            options=["Horizon", "Beam Width"], label="Wedge extends to"
        ),
        "theory_model": mo.ui.dropdown(
            ["EOS2021", "EOS2016Faint", "EOS2016Bright"],
            value="EOS2021",
            label="Theory Model: ",
        ),
    }
    sense_form = (
        mo.md("\n\n".join("{%s}" % key for key in sense_inputs))
        .batch(**sense_inputs)
        .form()
    )
    return sense_form, sense_inputs


@app.cell
def __(mo, observation_form, observatory_form, sense_form):
    form = mo.tabs(
        {
            "Observatory": observatory_form,
            "Observation": observation_form,
            "Sensitivity": sense_form,
        }
    )
    form
    return (form,)


@app.cell
def __(observatory_form, p21c, units):
    observatory = p21c.Observatory.from_profile(
        observatory_form.value["profile"],
        frequency=observatory_form.value["frequency"] * units.MHz,
    )
    return (observatory,)


@app.cell
def __(observation_form, observatory, p21c, units):
    observation = p21c.Observation(
        observatory=observatory,
        time_per_day=observation_form.value["time_per_day"] * units.hour,
        integration_time=observation_form.value["integration_time"] * units.s,
        n_channels=observation_form.value["n_channels"],
        bandwidth=observation_form.value["bandwidth"] * units.MHz,
        redundancy_tol=observation_form.value["redundancy_tol"],
        coherent=observation_form.value["coherent"],
        spectral_index=observation_form.value["spectral_index"],
        tsky_amplitude=observation_form.value["tsky_amplitude"] * units.K,
        tsky_ref_freq=observation_form.value["tsky_ref_freq"] * units.MHz,
    )
    return (observation,)


@app.cell
def __(littleh, observation, p21c, sense_form, units):
    sensitivity = p21c.PowerSpectrum(
        observation=observation,
        no_ns_baselines=sense_form.value["no_ns_baselines"],
        horizon_buffer=sense_form.value["horizon_buffer"] * littleh / units.Mpc,
        foreground_model="moderate"
        if sense_form.value["foreground_model"] == "Horizon"
        else "optimistic",
        theory_model=_ALL_THEORY_POWER_SPECTRA[sense_form.value["theory_model"]](),
    )
    return (sensitivity,)


@app.cell
def __(observatory, plt):
    plt.hist(observatory.baseline_lengths.flatten(), bins=40)
    plt.xlabel("Baseline Length (Wavelengths)")
    plt.ylabel("Number of baselines")
    plt.title("Histogram of Baseline Lengths")
    plt.gca()
    return


@app.cell
def __(observatory):
    observatory.baseline_lengths.unit
    return


@app.cell
def __():
    return


@app.cell
def __(observatory, plt):
    _red_bl = observatory.get_redundant_baselines()
    _baseline_group_coords = observatory.baseline_coords_from_groups(_red_bl)
    _baseline_group_counts = observatory.baseline_weights_from_groups(_red_bl)

    plt.figure(figsize=(7, 5))
    plt.scatter(
        _baseline_group_coords[:, 0],
        _baseline_group_coords[:, 1],
        c=_baseline_group_counts,
        s=5,
    )
    plt.xlim(
        _baseline_group_coords[:, :2].min().value,
        _baseline_group_coords[:, :2].max().value,
    )
    plt.ylim(
        _baseline_group_coords[:, :2].min().value,
        _baseline_group_coords[:, :2].max().value,
    )

    plt.xlabel(f"Baseline Length [{_baseline_group_coords.unit}]")
    plt.ylabel(f"Baseline Length [{_baseline_group_coords.unit}]")

    _cbar = plt.colorbar()
    _cbar.set_label("Number of baselines in group", fontsize=15)
    plt.tight_layout()

    plt.gca()
    return


@app.cell
def __(LogNorm, observation, observatory, plt):
    plt.imshow(
        observation.uv_coverage,
        extent=(observatory.ugrid().min(), observatory.ugrid().max()) * 2,
        norm=LogNorm(),
    )
    plt.xlabel("Baseline Length (Wavelengths)")
    plt.ylabel("Baseline Length (Wavelengths)")
    plt.title("UV Coverage")
    _cbar = plt.colorbar()
    _cbar.set_label("Effective # of Samples")
    plt.gca()
    return


@app.cell
def __(LogNorm, observation, plt):
    plt.imshow(
        observation.total_integration_time.to("hour").value,
        extent=(observation.ugrid.min(), observation.ugrid.max()) * 2,
        norm=LogNorm(),
    )
    plt.xlabel("Baseline Length (Wavelengths)")
    plt.ylabel("Baseline Length (Wavelengths)")
    plt.title("Total Integration Time")
    _cbar = plt.colorbar()
    _cbar.set_label("Total Integration Time [hours]")

    plt.gca()
    return


@app.cell
def __(sensitivity):
    sense2d = sensitivity.calculate_sensitivity_2d()
    return (sense2d,)


@app.cell
def __(plt, sense2d, sensitivity):
    sensitivity.plot_sense_2d(sense2d)
    plt.gca()
    return


@app.cell
def __(np, plt, sensitivity):
    plt.plot(
        sensitivity.k1d,
        sensitivity.calculate_sensitivity_1d(),
        label="sample+thermal",
        ls="-",
    )
    plt.plot(
        sensitivity.k1d,
        sensitivity.calculate_sensitivity_1d(sample=False),
        label="thermal",
        ls="--",
    )
    plt.plot(
        sensitivity.k1d,
        sensitivity.calculate_sensitivity_1d(thermal=False),
        label="sample",
        ls="--",
    )
    plt.plot(
        sensitivity.k1d,
        sensitivity.delta_squared,
        label=f"Theory ({sensitivity.theory_model.__class__.__name__})",
        color="k",
    )
    _mask = ~np.isinf(sensitivity.calculate_sensitivity_1d())
    plt.xlim(sensitivity.k1d[_mask].min().value, sensitivity.k1d[_mask].max().value)
    plt.title(f"Sensitivity at z={1420/sensitivity.frequency.to_value('MHz') - 1:.1f}.")
    plt.legend()
    plt.yscale("log")
    plt.xlabel("k [h/Mpc]")
    plt.ylabel(r"$\Delta^2_N\ \  [{\rm mK}^2]$")
    plt.gca()
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
