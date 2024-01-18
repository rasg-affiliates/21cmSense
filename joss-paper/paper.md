---
title: '21cmSense: A modular, open-source 21 cm sensitivity calculator'
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
    orcid: 0000-0000-0000-0000
  - name: Matthew Kolopanis
    affiliation: 3
    orcid: 0000-0000-0000-0000
affiliations:
 - name: Scuola Normale Superiore, Italy
   index: 1
 - name: Brown University, USA
   index: 2
 - name: Arizona State University, USA
   index: 3
date: 18 January 2024
bibliography: paper.bib
---

# Summary

The 21cm line of neutral hydrogen is a powerful probe of the high-redshift
universe, and is the subject of a number of current and upcoming
low-frequency radio experiments, including the MWA [@mwa], LOFAR [@lofar], HERA [@hera]
and the SKA [@Pritchard2015].
21cmSense is a Python package that provides a modular framework for calculating the
sensitivity of these experiments, in order to enhance the process of their design.
This paper presents version 2 of 21cmSense, which has been re-written from the ground up
to be more modular and extensible, and to provide a more user-friendly interface -- as
well as converting the well-used legacy package, presented in [@Pober2014] from Python 2 to 3.

21cmSense computes noise estimates under the framework of *map-making*, in which the
many baselines of an interferometer are binned into a UV grid before a Fourier Transform
over the frequency axis is performed. This is a common approach in the field, although
other approaches exist, such the delay-spectrum method [@Parsons2012].
The full sensitivity calculation in the map-making approach is rather involved and
computationally expensive in its most general form [@fhd], however 21cmSense uses a few
key assumptions to accelerate the calculation:

1. The UV grid is chosen to have cells that are comparable to the instrument's beam size.
   This maximizes UV-resolution while keeping the covariance between UV cells small
   (since the UV footprint of a visibility does not extend beyond the cell significantly).
   This removes the need for tracking the full covariance between cells, and also removes
   the need to perform a beam convolution, which can be expensive.
2. We do not consider flagging of visibilities due to RFI and other systematics, which
   can complicate the propagation of uncertainties.

Beyond these assumptions, there is also the current limitation that 21cmSense computes
the sensitivity under the map-making framework. Nevertheless, the modularity included
in this new version provides a path forward to include delay-spectrum calculations in
the future.

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
   classic sensitivity of the (map-making style) power spectrum. However, the framework
   can be extended to other summaries, for example wavelets [@Trott2016a].
6. Improved speed: the new version of 21cmSense is significantly faster than the legacy
   version, due to a number of vectorization improvements in the code.
7. Built-in profiles for several major experiments: MWA, HERA and SKA-1. These can be
   used as-is, or as a starting point for defining a custom instrument.



# Statement of need

`21cmSense` provides a simple interface for computing the expected sensitivity of
radio interferometers that aim to measure the 21cm line of neutral hydrogen.
This field is growing rapidly, with a number of experiments currently underway or in the planning stages.
Historically, `21cmSense` has been a trusted tool for the design of these experiments [@Pober2013; @Pober2014; @Greig2020] and for forecasting parameter constraints [@Greig2015; @Greig2017; @Greig2018].
This overhauled, modularized version of `21cmSense` provides a more user-friendly interface, improved performance, and the extensibility required for the next generation, as evidenced by its usage in the
literature [@Brietman2024,@Schosser2024].

# Acknowledgements

We acknowledge helpful conversations with Danny Jacobs.

# References
