.. _jax-performance:

JAX GPU performance
===================

These measurements use an NVIDIA RTX 5090 in FP64. The official-example scope
on this page means Python examples present in ``hiddenSymmetries/simsopt``
``master`` at commit ``9e027eac38028d57aa23777be52a781aa860e347``.
That source identity does not make a locally built native extension an official
upstream binary. Workload, source, solver policy, timing boundary, and native
comparator matter for every ratio below.

Official upstream example coverage
----------------------------------

The 2026-09-22 UTC verification ran all 25 eligible examples mapped to that
upstream commit at their shipped/default work budgets on validation snapshot
``998148365e8db55577b4372c4c791ef6bc20cc35``. These are separate native/JAX
CPU and native/JAX GPU comparisons, not the full upstream example catalog or
a complete three-lane comparison matrix.

.. list-table::
   :header-rows: 1

   * - Comparison
     - Case-contract pass
     - Qualified quality-band
     - Failed
   * - Native / JAX CPU
     - 20
     - 5
     - 0
   * - Native / JAX GPU
     - 19
     - 4
     - 2

The four qualified GPU cases are BoozerQA, coil forces, finite-build stage two,
and minimal stage two. Coil forces retains its raw GPU solver failure. QFM's
GPU endpoint-quality failure and the NCSX GPU long-trajectory comparison
failure remain unresolved. Passing a case contract or an engineering band
does not by itself prove convergence.

The native/JAX pairs use a source-bound branch-native build. A separate,
independently built official reference checks selected construction, input
and count quantities; it reports endpoints without adjudicating them. The
individual case contracts judge branch-lane endpoints. This does not prove
equivalence of every official trajectory or shipped setting. PM4Stell's
backtracking deviation and the planar-coil upstream-gradient limitation remain
disclosed. The NCSX contract admits small hit-count differences separately
from its endpoint check.

The branch-native build still differs from official upstream: 33 C++ source
files differ, although the current GSCO kernel differs only in an include and
the Biot--Savart kernel retains the older SIMD lane-storage interface. Four
native scripts explicitly fix the surface degrees of freedom: standard stage
two, planar stage two, coil forces, and permanent-magnet QA. The timing rows
below identify their earlier comparators separately; a source-bound local
build is not an official upstream binary.

Reports and raw packets are retained locally in
``.artifacts/reconciliation-execution-20260921/`` and are not distributed in
Git. The counts describe that frozen snapshot. Verification jobs overlapped,
so their elapsed times provide no performance result. The measurements below
retain their own earlier source revisions and timing boundaries.

Measured official-example computational workflows
-------------------------------------------------

The following measurements use mirror workloads mapped to official examples,
with the branch-built native comparator described above. They are not timings
of the public example command from import through report output. Warm means
repeated execution in one process after compilation; cold means the first
execution with an empty compilation cache. The ratios divide native wall time
by synchronized GPU wall time. Each timing campaign has three warm samples;
these numbers do not establish performance on other hardware or a newer
source revision.

.. list-table::
   :header-rows: 1
   :widths: 22 19 20 39

   * - Example/workload
     - Warm native/GPU
     - First-run native/GPU
     - Scope and comparator
   * - GSCO multistep (``wireframe_gsco_multistep.py``)
     - About 4.0x
     - About 3.2x, empty cache
     - Common host assembly, response setup, solve, and synchronized
       extraction; branch source ``58a9d103f``. Native used eight host
       threads, selected from one- and eight-thread pilots. Native GSCO has
       the branch undo-index fix.
   * - Standard stage two (``stage_two_optimization.py``)
     - About 1.5x
     - 1.20x with populated cache; 0.52x with empty cache
     - Matched computational workflow at ``674bafd2b``; branch native at
       eight OpenMP threads versus GPU at one host thread.
   * - Planar stage two (``stage_two_optimization_planar_coils.py``)
     - About 1.4x
     - 1.00x with populated cache; 0.41x with empty cache
     - Same timing boundary and comparator as standard stage two; the
       branch native planar derivative cache differs from upstream.

The GSCO protocol includes construction and response setup through solve and
synchronized extraction, but excludes imports, provenance, and report
serialization. It verified 30 native/GPU pairings and 69 comparisons per
pairing. The public fresh-process GSCO outputs differed in work, so they do
not supply a matched public-command speedup.

The stage-two protocol includes input construction, initial evaluation,
Taylor check, both host SciPy L-BFGS-B stages, final evaluation, and readback.
It excludes imports, JAX startup, provenance, and report serialization.
Its native eight-thread setting was the better of one- and eight-thread pilots,
not an exhaustive native-thread optimum. The same optimizer policy and nominal
800-iteration budget were used, but evaluations differed: 981--1268 native
versus 1102 GPU for standard stage two, and 839--874 native versus 867 GPU for
planar stage two. One native planar warm repetition stopped after 776 total
iterations; the other planar repetitions reached the cap. These are wall-time
measurements of the bounded solves, not time-to-convergence measurements.
The 30 comparison pairings for each stage-two case passed their declared
checks. The retained timing authority is ``674bafd2b``, not the later
precision head. See ``.artifacts/upstream-mirror-execution-20260919/``
``ALL-MIRROR-RESULTS.md`` and ``FINAL-TIMING-CLAUDE-REVIEW.md`` for receipts
and timing boundaries.

Historical official-example measurements
----------------------------------------

These workloads have official upstream counterparts, but the measurements use
older branch harnesses, sizes or solver policies. They do not qualify the
current shipped/default workflow. The two newer stage-two rows above use a
broader timing boundary than the minimize-region rows here.

.. list-table::
   :header-rows: 1
   :widths: 35 21 21 23

   * - Historical workload and timed boundary
     - GPU
     - Branch native
     - Reported ratio / qualification
   * - PM4Stell, nphi=64 (``permanent_magnet_PM4Stell.py``)
     - 7.98 s warm; 9.66 s cold
     - 16.12 s at 32 threads
     - 2.02x reported; placements bitwise identical in 20 comparisons
   * - Standard / planar stage-two matched minimize regions
     - 3.18 s / 3.21 s
     - 10.99 s at 8 threads / 10.70 s at 16 threads
     - 3.46x / 3.33x inherited measurements; different timing protocol
       from the newer stage-two rows above
   * - Stochastic stage two, mc10 / mc400
     - 23.60 s / 24.18 s
     - 27.25 s / 28.80 s at 16 threads
     - 1.15x / 1.19x reported for the historical matched-state evaluator
   * - Coil forces, matched minimize region, 400+400 iterations
     - Not retained as an absolute time here
     - Eight OpenMP threads
     - 11.1x reported on 2026-09-15 with L-BFGS-B ``maxls=32``

The historical coil-forces run used ``maxls=32`` on both lanes; the current
example uses upstream ``maxls=20``. Its ratio is neither current-policy performance
nor whole-workflow acceleration. The September 19 shipped/default packet
failed the GPU work-budget admission for coil forces and a shared
objective-quality gate for PM4Stell. The later verification above classifies
coil forces as quality-band on both pairs, retaining its raw GPU failure,
and PM4Stell as a case-contract pass on both pairs with its disclosed
backtracking deviation. Neither
update recertifies these historical timings. A historical GPU strict
regression collection recorded 177 passes and zero failures on 2026-09-15;
it does not qualify the present tree.

Historical workload-size observations
-------------------------------------

In earlier local sweeps, a GSCO 48x50 wireframe roughly tied within 30%,
while a 96x100 case measured 5.1x and 4.2x at bitwise-identical currents. RCLS dense solves through n=1040 measured
0.25x--0.58x of native speed on the GPU. These size-dependent observations
are historical and do not establish a current crossover or a public-command
speedup.

Other older example observations include a full-K MUSE GPU ratio of 1.8x
(about 2.4x projected at matched work), while the QA permanent-magnet
variant differed by 9.3% in dipole moments at nphi=64 under the same
algorithm and stopping rule. A later relax-and-split QA protocol measured
maximum absolute difference 1.05e-12 after the stacked-predicate fix
``4c75551ab``. The older ``boozerQA.py`` value-plus-gradient evaluation
measured 2.6 s GPU versus 22 ms native, and ``boozer.py`` requested 243 GiB
of device memory at mpol 16 / 48x48. These observations use different
workloads and policies from the retained 25-case mirror and are not present
performance recommendations. The retained shipped/default checks for Boozer,
BoozerQA, QFM and the tracing cases fail; older measurements cannot override
those outcomes.

The JAX backend setup and runtime modes are described on the ``jax`` backend
page.
