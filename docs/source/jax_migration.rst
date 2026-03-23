JAX Backend Migration Guide
===========================

This page documents the mapping between CPU (simsoptpp) APIs and their
JAX equivalents for callers migrating to the JAX code path.

The JAX modules do **not** replace the CPU modules.  Both coexist.
The CPU path remains the default and the correctness oracle.

As of the JAX ``0.9.2`` runtime, the trusted least-squares backend remains
``optimizer_backend="scipy"``.  The private ``hybrid`` and ``ondevice``
backends now target the same runtime, but they remain a separate validation
track because they still depend on private optimizer internals.

Stage 2 (Field + Flux Objective)
--------------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 40 20

   * - CPU API
     - JAX Equivalent
     - Module
   * - ``BiotSavart(coils)``
     - ``BiotSavartJAX(coils)``
     - ``simsopt.field``
   * - ``bs.B()``
     - ``bs_jax.B()``
     - same API
   * - ``bs.dB_by_dX()``
     - ``bs_jax.dB_by_dX()``
     - same API
   * - ``bs.B_vjp(v)``
     - ``bs_jax.B_vjp(v)``
     - same API
   * - ``SquaredFlux(surf, bs)``
     - ``SquaredFluxJAX(surf, bs_jax)``
     - ``simsopt.objectives``
   * - ``sopp.integral_BdotN(…)``
     - ``integral_BdotN_jax(…)``
     - ``simsopt.objectives.integral_bdotn_jax``

Single-Stage (Boozer Solver + Objectives)
-----------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 40 20

   * - CPU API
     - JAX Equivalent
     - Module
   * - ``BoozerSurface(bs, surf, label, target, cw)``
     - ``BoozerSurfaceJAX(bs_jax, surf, label, target, cw)``
     - ``simsopt.geo``
   * - ``booz.run_code(iota, G)``
     - ``booz_jax.run_code(iota, G)``
     - same API
   * - ``BoozerResidual(booz, bs)``
     - ``BoozerResidualJAX(booz_jax, bs_jax)``
     - ``simsopt.geo``
   * - ``Iotas(booz)``
     - ``IotasJAX(booz_jax)``
     - ``simsopt.geo``
   * - ``NonQuasiSymmetricRatio(booz, bs)``
     - ``NonQuasiSymmetricRatioJAX(booz_jax, bs_jax)``
     - ``simsopt.geo``

Internal APIs (No Direct Replacement Needed)
---------------------------------------------

These simsoptpp internals are replaced by JAX autodiff and do not need
to be called directly:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - CPU Internal
     - JAX Replacement
   * - ``sopp.biot_savart_vjp_graph(…)``
     - ``jax.vjp`` through ``biot_savart_B``
   * - ``boozer_surface_residual_dB(…)``
     - ``jax.grad`` through ``boozer_residual_scalar``
   * - ``boozer_surface_dlsqgrad_dcoils_vjp(…)``
     - ``_boozer_ls_coil_vjp()`` via ``jax.vjp``
   * - ``boozer_surface_dexactresidual_dcoils_dcurrents_vjp(…)``
     - ``_boozer_exact_coil_vjp()`` via ``jax.vjp``

VJP Calling Convention Change
-----------------------------

The JAX VJP hooks stored in ``res['vjp']`` have signature
``(lm, booz_surf, iota, G)``, **not** the CPU signature ``(lm, booz_surf)``.
This is because JAX VJPs construct the decision vector from explicit args
rather than reading ``booz_surf`` internal state.

What Is NOT Changing
--------------------

- The ``Optimizable`` dependency graph and ``need_to_run_code`` semantics
  are preserved.
- ``Coil``, ``Current``, ``CurveXYZFourier`` and all curve classes remain
  CPU-only.  The JAX path reads coil geometry from these objects at the
  ``run_code()`` boundary.
- MPI support is not part of the JAX lane (v1 is single-process).
- Field-line tracing (``simsopt.field.tracing``) is not ported.
