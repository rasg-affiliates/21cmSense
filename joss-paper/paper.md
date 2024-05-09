---
title: '21cmSense v2: A modular, open-source 21 cm sensitivity calculator'
tags:
  - Python
  - astronomy
  - 21 cm Cosmology
authors:
  - name: Steven G. Murray
    orcid: 0000-0003-3059-3823
    equal-contrib: true
    affiliation: 1 # (Multiple affiliations must be quoted)
    corresponding: true
  - name: Jonathan Pober
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 2
    orcid: 0000-0002-3492-0433
  - name: Matthew Kolopanis
    affiliation: 3
    orcid: 0000-0002-2950-2974
affiliations:
 - name: Scuola Normale Superiore, Italy
   index: 1
 - name: Department of Physics, Brown University, Providence, RI, USA
   index: 2
 - name: School of Earth and Space Exploration, Arizona State University, Tempe, AZ, USA
   index: 3
date: 18 January 2024
bibliography: paper.bib
---

# Summary

The 21cm line of neutral hydrogen is a powerful probe of the high-redshift
universe (Cosmic Dawn and the Epoch of Reionization), with an unprecedented potential to
inform us about key processes of early galaxy formation, the first stars and even
cosmology and structure formation [@Liu2020], via intensity mapping.
It is the subject of a number of current and upcoming
low-frequency radio experiments, including the MWA [@mwa], LOFAR [@lofar], HERA [@hera]
and the SKA [@Pritchard2015], which complement the detailed information concerning the
brightest sources in these early epochs from powerful optical and near-infrared telescopes
such as the JWST [@jwst].


21cmSense is a Python package that provides a modular framework for calculating the
sensitivity of these experiments, in order to enhance the process of their design.
This paper presents version v2.0.0 of 21cmSense, which has been re-written from the ground up
to be more modular and extensible, and to provide a more user-friendly interface -- as
well as converting the well-used legacy package, presented in [@Pober2013; @Pober2014] from Python 2 to 3.

21cmSense can compute sensitivity estimates for both map-making [@fhd] and
delay-spectrum [@Parsons2012] approaches to power-spectrum estimation.
The full sensitivity calculation is rather involved and
computationally expensive in its most general form, however 21cmSense uses a few
key assumptions to accelerate the calculation:

1. Each baseline (pair of antennas) in the interferometer intrinsically measures a dense
   blob of 2D spatial Fourier modes of the sky intensity distribution, centred at a
   particular Fourier  coordinate $(u,v)$ given by the displacement vector between the
   antennas forming the baseline, and covering an area in this $(u,v)$-space that is given
   by the Fourier-transform of the primary beam of the instrument.
   The Fourier-space representation of the sky is thus
   built up by collecting many baselines that cover the so-called "$(u,v)$-plane".
   ``21cmSense`` approximates this process of synthesising many baselines by
   nearest-grid-point interpolation onto a regular grid in the $(u,v)$-plane.
   Furthermore, importantly the $(u,v)$-grid is chosen to have cells that are comparable
   to the instrument's Fourier-space beam size, so that a particular baseline essentially
   measures a single cell in the grid, and no more.
   This maximizes resolution while keeping the covariance between cells small.
   This removes the need for tracking the full covariance between cells, and also removes
   the need to perform a beam convolution, which can be expensive.
2. We do not consider flagging of visibilities due to RFI and other systematics, which
   can complicate the propagation of uncertainties.

Some of the key new features introduced in this version of 21cmSense include:

1. Simplified, modular library API: the calculation has been split into modules that can
   be used independently (for example, a class defining the `Observatory`, the
   `Observation` and the `Sensitivity`). These can be used interactively via Jupyter
   [@jupyter] or other interactive interfaces for Python, or called as library functions
   in other code.
2. Command-line interface: the library can be called from the command-line, allowing
   for easy scripting and automation of sensitivity calculations.
3. More accurate cosmological calculations using `astropy` [@Robitaille2013; @astropy]
4. Improved documentation and examples, including a Jupyter notebook that walks through
   the calculation step-by-step.
5. Generalization of the sensitivity calculation. The `Sensitivity` class is an abstract
   class from which the sensitivity of differing summary statistics can be defined.
   Currently, its only implementation is the `PowerSpectrum` class, which computes the
   classic sensitivity of the power spectrum. However, the framework
   can be extended to other summaries, for example wavelets [@Trott2016a].
6. Improved speed: the new version of 21cmSense is significantly faster than the legacy
   version, due to a number of vectorization improvements in the code.
7. Built-in profiles for several major experiments: MWA, HERA and SKA-1. These can be
   used as-is, or as a starting point for defining a custom instrument.

An example of the predicted sensitivity of the HERA experiment after a year's observation
at $z=8.5$ is shown in Figure \ref{sense}, corresponding to the sampling of the $(u,v)$-grid
shown in Figure \ref{uvsampling}. The sensivity here is represented as a "noise power"
(i.e. the contribution to the power spectrum from thermal noise).
This figure also demonstrates that the new
21cmSense is capable of producing sensitivity predictions in the cylindrically-averaged
2D power spectrum space, which is helpful for upcoming experiments.

![Sampling of the $(u,v)$-plane for the HERA experiment during a full year of observations.\label{uvsampling}](uv-sampling.pdf)

![Predicted noise-power of 1000 hours (one year) of HERA observations, as a function of perpendicular and line-of-sight fourier scale. The noise-power is represented for each $k$-model.\label{sense}](2dps.pdf)

# Statement of need

`21cmSense` provides a simple interface for computing the expected sensitivity of
radio interferometers that aim to measure the 21cm line of neutral hydrogen.
This field is growing rapidly, with a number of experiments currently underway or in
the planning stages. Historically, `21cmSense` has been a trusted tool for the design of
these experiments [@Pober2013; @Pober2014; @Greig2020] and for forecasting parameter
constraints [@Greig2015; @Greig2017; @Greig2018].
This overhauled, modularized version of `21cmSense` provides a more user-friendly
interface, improved performance, and the extensibility required for the next generation,
as evidenced by its usage in the literature [@Breitman2024; @Schosser2024].

# Acknowledgements

We acknowledge helpful conversations with Danny Jacobs.

# References
