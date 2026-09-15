.. _jax-performance:

JAX GPU performance
===================

This page reports measured performance of the JAX GPU backend against the
native CPU backend on an NVIDIA RTX 5090 accelerator in FP64. Each row below
names the workload it covers and the parity check that was applied. The
numbers do not transfer to other resolutions, grid sizes, or problems; see
`When to use the GPU backend`_ for the measured crossover points.

What was measured and how
-------------------------

JAX timings are reported two ways. *Warm* timings reuse a persistent
compilation cache and represent steady-state execution after the compiled
executables are cached. *Cold* timings use a fresh compilation cache and
include compilation time. The first call to a compiled function is the slow
one; later synchronized calls are timed separately.

Native timings are the best result of an OpenMP thread-count sweep (4, 8, 16,
and 32 threads, best value kept per workload). Comparisons run native and JAX
lanes back to back at matched solver policy and matched work. Speedups are
native-seconds-over-JAX-seconds ratios: the median of paired per-run ratios
for the single-stage rows, and ratios of lane medians elsewhere.

Results
-------

.. list-table::
   :header-rows: 1
   :widths: 26 12 12 12 22 14

   * - Workload (scope)
     - JAX warm
     - Native best
     - Speedup
     - Parity
     - Packet (date)
   * - Nested least-squares Boozer solve, NCSX 48x48, unguarded algorithm
       (``tests/geo/test_nested_ls_ncsx.py``)
     - 57.0 s (97.4 s cold)
     - 830.4 s OpenMP (1832.7 s single-threaded upstream kernel)
     - 14.6x (32x vs single-threaded upstream)
     - Native/JAX solve traces compared; identical algorithm on both lanes
     - pass 4 (2026-09-14)
   * - Nested least-squares Boozer solve, NCSX 48x48, divergence guard on
       both lanes (``tests/geo/test_nested_ls_ncsx.py``)
     - 35.04 s (80.74 s cold)
     - 286.28 s OpenMP (530.21 s single-threaded upstream kernel)
     - 8.2x warm, 3.5x cold (15.1x warm, 6.6x cold vs single-threaded upstream)
     - Native/JAX solve traces compared; identical algorithm on both lanes
     - pass 5 (2026-09-15), code as shipped
   * - Exact-constraint single-stage optimization, 1000 iterations (``tests/geo/test_single_stage_exact_analytic.py``)
     - 24.8 s
     - 55.7 s process wall
     - 2.17x (pass 5 measured 2.24x)
     - Identical initial state; gradient relative L2 difference 3.9e-13
     - pass 5b (sealed 2026-09-15 13:03 UTC)
   * - Shipped single-stage vacuum example (``examples/3_Advanced/single_stage_boozer_vacuum_optimization.py``)
     - n/a
     - n/a
     - 2.20x (pass 5 measured 2.30x)
     - Initial-gradient absolute difference 8.8e-16; final objectives 4.3058761e-08 (native) vs 4.3821759e-08 (JAX)
     - pass 5b (sealed 2026-09-15 13:03 UTC)
   * - PM4Stell permanent magnets, nphi=64 (``examples/2_Intermediate/permanent_magnet_PM4Stell.py``)
     - 7.98 s (9.66 s cold)
     - 16.12 s (32 threads)
     - 2.02x
     - Magnet placements bitwise identical (maximum ULP 0 over 20 comparisons)
     - pass 5 (2026-09-15)
   * - Stage-two coil optimization (``examples/2_Intermediate/stage_two_optimization.py``)
     - 3.18 s
     - 10.99 s (8 threads)
     - 3.46x
     - Compared over a matched minimize region
     - pass 5b (2026-09-15), inherited
   * - Planar-coil stage-two optimization (``examples/2_Intermediate/stage_two_optimization_planar_coils.py``)
     - 3.21 s
     - 10.70 s (16 threads)
     - 3.33x
     - Compared over a matched minimize region
     - pass 5b (2026-09-15), inherited
   * - Stochastic stage-two optimization, mc10 / mc400 (``examples/2_Intermediate/stage_two_optimization_stochastic.py``)
     - 23.60 s / 24.18 s
     - 27.25 s / 28.80 s (16 threads)
     - 1.15x / 1.19x
     - Matched-state evaluator parity: objective absolute difference <= 4.8e-20, gradient maximum absolute difference <= 1.6e-16
     - pass 5 (2026-09-15)
   * - GPU strict regression collection
     - n/a
     - n/a
     - n/a
     - 177 passed, 0 failed
     - pass 5b (sealed 2026-09-15 13:03 UTC)

Commit 28b30477d adds the same divergence guard to both lanes: the LS-Newton
inner solve stops once the gradient norm exceeds 1e3 times its entry value on
rejected line-search trial points instead of running to the 40-step cap. The
guard removes wasted Hessian assemblies that cost the single-threaded native
kernel about 10 s each and the GPU about 0.1 s, so both lanes get faster and
the ratio drops. The headline number for the shipped code is 8.2x warm (3.5x
cold) against the OpenMP kernel; the unguarded row is kept so readers can
compare with the upstream algorithm as released.

All times are medians of repeated runs. The coil-forces example
(``examples/3_Advanced/coil_forces.py``) at matched policy (L-BFGS-B maxls 32
on both lanes, 400+400 stage iterations) measures 11.1x on the minimize-region
clock against native at its best OpenMP count (8) (pass 5b, sealed 2026-09-15
13:03 UTC).

When to use the GPU backend
---------------------------

The GPU backend wins once a workload is large enough to amortize compilation
and dispatch overhead, and loses below that point:

* Nested least-squares Boozer solves: use the GPU from 32x32 / mpol 10
  upward (14.2 s vs 68.5 s outer-solve scope at 32x32). At 18x18 / mpol 6
  and below the native CPU backend is faster (5.8 s vs 8.8 s), so stay on
  the CPU.
* GSCO wireframes (``examples/2_Intermediate/wireframe_gsco_modular.py``):
  the shipped 48x50 problem ties within +/-30%, so either backend is fine.
  At 96x100 the GPU wins with speedups of 5.1x and 4.2x at bitwise-identical
  currents.
* RCLS dense wireframe solves
  (``examples/2_Intermediate/wireframe_rcls_basic.py``): stay on the CPU
  through problem size n=1040, where the GPU reaches only 0.25x--0.58x of
  native speed.
* Permanent magnets: the shipped full-K MUSE problem
  (``examples/2_Intermediate/permanent_magnet_MUSE.py``) runs 1.8x faster on
  the GPU (about 2.4x at matched work, projected). The QA variant
  (``examples/2_Intermediate/permanent_magnet_QA.py``) is not a GPU
  candidate: its dipole moments differ by 9.3% relative at nphi 64 under the
  same algorithm and stopping rule, which is a parity failure. A relax-and-split
  protocol for the QA variant reaches GPU-vs-native parity of maximum absolute
  difference 1.05e-12 after the stacked-predicate fix (4c75551ab) (pass 5b,
  sealed 2026-09-15 13:03 UTC).
* Boozer-surface value-and-gradient examples
  (``examples/2_Intermediate/boozerQA.py``,
  ``examples/2_Intermediate/boozer.py``) are not GPU candidates. The shipped
  ``boozerQA`` value-plus-gradient evaluation runs about 118x slower on the
  GPU than on native CPU (2.6 s vs 22 ms) because an exact-Newton inner solve
  sits inside the compiled region, and ``boozer.py`` exceeds device memory
  (243 GiB requested) at mpol 16 / 48x48.

The JAX backend setup, runtime modes, and migration path are described on
the ``jax`` backend page.
