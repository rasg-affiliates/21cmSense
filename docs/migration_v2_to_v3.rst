========================
Migrating from v2 to v3
========================

v3 contains a significant internal rewrite aimed at making sensitivity calculations
over many frequencies/redshifts much faster and more memory-efficient. The core change
underlying almost everything below is that **frequency is no longer "baked into"**
``PrimaryBeam``/``Observatory`` at construction time. Previously, computing a sensitivity
at a new frequency meant cloning a new ``Beam`` -> ``Observatory`` -> ``Observation``
chain, and re-doing the (expensive) antenna-baseline and redundancy calculations from
scratch every time. Now, ``Observatory`` (and the antenna positions/redundancy
information it owns) is frequency-independent and reusable across a whole frequency
sweep; ``frequency`` lives on ``Observation``, and is passed explicitly to the handful of
``Beam``/``Observatory`` methods that need it.

This is a breaking change to the public API. The changes
below are exhaustive; if your code only uses ``Observatory``, ``Observation``,
``Sensitivity``/``PowerSpectrum`` and their YAML config files at a "black box" level (i.e.
you build them and call ``calculate_sensitivity_2d`` or similar), you likely need no
changes beyond moving ``frequency`` from your beam config to your observation config (see
below).

Frequency moved from ``PrimaryBeam``/``Observatory`` to ``Observation``
=========================================================================

``PrimaryBeam`` (and subclasses like ``GaussianBeam``) no longer take ``frequency`` as a
constructor argument, and ``Observatory`` no longer exposes a ``frequency`` property.
Instead, ``Observation`` has a ``frequency`` field (default ``150 * un.MHz``).

.. code-block:: python

    # v2
    beam = GaussianBeam(frequency=150 * un.MHz, dish_size=14 * un.m)
    observatory = Observatory(antpos=antpos, beam=beam)
    observation = Observation(observatory=observatory)

    # v3
    beam = GaussianBeam(dish_size=14 * un.m)
    observatory = Observatory(antpos=antpos, beam=beam)
    observation = Observation(observatory=observatory, frequency=150 * un.MHz)

If you are loading a YAML config for the observatory/beam, remove ``frequency`` from the
``beam:`` block and add it to the observation config instead (or pass it to
``Observation(..., frequency=...)`` directly).

``Sensitivity.at_frequency`` still works the same way from the caller's perspective --
it now clones the ``Observation`` (rather than the ``Beam``) at the new frequency.

Beam quantities are now methods, not properties
==================================================

Because a ``PrimaryBeam`` no longer has a fixed frequency, everything on it that used to
depend on ``self.frequency`` is now a **method** that takes ``frequency`` as an argument,
rather than a zero-argument ``@property``. This affects ``area``, ``width``,
``first_null``, ``sq_area``, ``uv_resolution``, ``fwhm``, ``b_eff``, and (for
``GaussianBeam``) ``dish_size_in_lambda``.

.. code-block:: python

    # v2
    beam.area
    beam.fwhm
    beam.b_eff

    # v3
    beam.area(frequency)
    beam.fwhm(frequency)
    beam.b_eff(frequency)

``PrimaryBeam.at(frequency)`` is removed -- there is no longer a frequency-specific beam
object to construct; just pass ``frequency`` to the method you need.

``Observatory`` baseline API changes
=======================================

Several ``Observatory`` baseline-related attributes and methods changed shape, units, or
signature as part of moving to a de-duplicated, half-plane baseline representation
(instead of the full ``(Nant, Nant, 3)`` matrix) for memory/speed reasons.

``baselines_metres`` is deprecated
-------------------------------------

``baselines_metres`` still exists but now emits a ``DeprecationWarning`` and returns the
same array as the new ``baselines`` attribute -- **not** the old ``(Nant, Nant, 3)``
matrix. Update code to use ``baselines`` (post baseline-filter) or
``unfiltered_baselines`` (pre baseline-filter) directly; both have shape ``(Nbls, 3)``.

.. code-block:: python

    # v2 -- shape (Nant, Nant, 3), includes every ordered antenna pair
    observatory.baselines_metres

    # v3 -- shape (Nbls, 3), one row per unordered antenna pair (post-filtering)
    observatory.baselines

``shortest_baseline`` / ``longest_baseline`` -> ``bl_min`` / ``bl_max``
---------------------------------------------------------------------------

These are renamed *and* now return **metres** instead of **wavelengths** (since
``Observatory`` no longer has a fixed frequency to convert with), and are computed from
the redundant baseline groups rather than every individual baseline pair (the min/max are
the same either way).

.. code-block:: python

    # v2 (wavelengths, at beam.frequency)
    observatory.shortest_baseline
    observatory.longest_baseline

    # v3 (metres)
    observatory.bl_min
    observatory.bl_max

If you need the old wavelength values, convert explicitly at your observation frequency,
e.g. ``observatory.bl_min * frequency / const.c``.

``get_redundant_baselines`` signature and return type changed
-------------------------------------------------------------------

``baseline_filters`` and ``ndecimals`` are no longer arguments to this method -- they are
now constructor fields on ``Observatory`` itself (``baseline_filters``, ``redundancy_tol``).
The return type changed from a ``dict`` keyed by ``(u, v, |u|)`` co-ordinates to a plain
``list`` of lists of integer indices into ``observatory.baselines``.

.. code-block:: python

    # v2
    groups = observatory.get_redundant_baselines(baseline_filters=my_filter, ndecimals=1)
    # groups: dict[(u, v, len), list[(i, j)]]

    # v3
    observatory = Observatory(..., baseline_filters=(my_filter,), redundancy_tol=1)
    groups = observatory.get_redundant_baselines()  # equivalently: observatory.redundant_baseline_groups
    # groups: list[list[int]], indices into observatory.baselines

The corresponding coordinate/weight helpers changed accordingly:

* ``Observatory.baseline_coords_from_groups(groups)`` (staticmethod) is removed --
  use ``observatory.redundant_baseline_vectors`` instead.
* ``Observatory.baseline_weights_from_groups(groups)`` (staticmethod) is removed --
  use ``observatory.redundant_baseline_weights`` instead.

``ugrid``/``ugrid_edges`` now take ``frequency``, not ``bl_max``
------------------------------------------------------------------

.. code-block:: python

    # v2
    observatory.ugrid(bl_max=200 * un.m)

    # v3
    observatory.ugrid(frequency=150 * un.MHz)

The grid now always extends to ``observatory.bl_max`` (there's no way to truncate it to a
smaller ``bl_max`` directly any more); use a baseline filter on the ``Observatory`` if you
need a shorter maximum baseline. Grid *positions* in metres are available via the new
``xgrid``/``xgrid_edges`` properties, which don't need a frequency at all.

``grid_baselines`` signature changed
----------------------------------------

``baselines``, ``weights``, ``baseline_filters`` and ``ndecimals`` are no longer
arguments -- gridding always uses the ``Observatory``'s own (filtered, redundancy-grouped)
baselines. A new ``frequency`` argument controls whether gridding happens in metres
(``frequency=None``, the default -- same grid reused across all frequencies, faster) or
wavelengths at a specific frequency (matching the old behaviour more closely). A new
``max_chunk_mem_gb`` argument controls memory use for large arrays.

``Observatory.from_yaml`` / ``from_profile`` / ``from_ska`` drop ``frequency``
===================================================================================

These classmethods no longer accept a ``frequency`` keyword argument, since
``Observatory`` (and its beam) no longer stores a frequency. Set ``frequency`` on the
``Observation`` instead.

``Observation`` field changes
=================================

* ``baseline_filters`` and ``redundancy_tol`` are no longer ``Observation`` constructor
  arguments -- pass them to ``Observatory`` instead (see above).
* ``frequency`` is now a genuine ``Observation`` constructor field (see the first
  section above), rather than a read-only property derived from the beam.
* ``baseline_groups``, ``baseline_group_coords``, ``baseline_group_counts``,
  ``baseline_group_lengths``, ``bl_min`` and ``bl_max`` are removed from ``Observation``.
  The equivalents now live on ``Observation.observatory``:
  ``redundant_baseline_groups``, ``redundant_baseline_vectors``,
  ``redundant_baseline_weights``, ``baseline_group_lengths``, ``bl_min``, ``bl_max``.

``Sensitivity``/``PowerSpectrum`` signature changes
=======================================================

``thermal_noise`` and ``sample_noise`` now take a single ``k`` (the magnitude) instead of
separate ``k_par``/``k_perp`` components:

.. code-block:: python

    # v2
    ps.thermal_noise(k_par, k_perp, trms)
    ps.sample_noise(k_par, k_perp)

    # v3
    import numpy as np
    k = np.sqrt(k_par**2 + k_perp**2)
    ps.thermal_noise(k, trms)
    ps.sample_noise(k)

Minor / lower-risk changes
==============================

* ``conversions.f2z`` now returns an (dimensionless) ``astropy.Quantity`` or array rather
  than always coercing to a Python ``float``. Wrap in ``float(...)`` if you need a plain
  scalar and pass a scalar frequency.
* ``conversions.dL_dth``, ``dL_df``, ``dk_du``, ``dk_deta`` and ``X2Y`` gained a
  ``with_h: bool = True`` keyword argument for returning results without factors of
  *little h*; the default preserves v2 behaviour.
