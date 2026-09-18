"""VMEC-free single-stage optimization on the flat coupled formulation.

This is the official example for the flat coupled single-stage problem
statement: coil degrees of freedom, vessel degrees of freedom and Boozer
surface degrees of freedom are ONE state vector, and there is no nested
equilibrium solve anywhere inside an objective evaluation.

The mirror lesson ``3_Advanced/single_stage_boozer_vacuum_optimization`` nests
two problems -- an outer optimizer moves coils, and every outer evaluation
runs an inner Newton solve to put the surface back on the Boozer manifold.
The flat formulation removes that inner solve entirely.  The rotational
transform and the net poloidal current are not solved for; they are closed in
closed form by a two-column least-squares solve of the Boozer system, which is
differentiable like everything around it.  What is left is a single scalar
objective over the whole state vector.

WHY THAT MATTERS FOR EXECUTION
------------------------------
Because there is no inner solve, the entire objective is one device program,
and the whole optimization is the fused on-device L-BFGS-B lane end to end --
no host round trip per accepted step, no host-side rejection or anchor
protocol.  This script publishes its own proof of that: it solves inside
``jax.transfer_guard("disallow")`` with ``host_transfer_audit()`` open and
reports the transfer ledger as an observable.  ``host_step_transfers`` and
``host_callback_transfers`` are zero on a fused run; a stepwise fallback would
make one of them nonzero, and the endpoint read-back after the guarded region
is the positive control that the audit saw anything at all.

THE LAYOUT IS FIXED AT 11 + 3 + 661
-----------------------------------
Eleven coil owner degrees of freedom (one free current plus a
curve-on-winding-surface family), three vessel degrees of freedom, and 661
boundary degrees of freedom -- a stellarator-symmetric
``SurfaceXYZTensorFourier`` at ``mpol = ntor = 10``.  The constructor
:func:`~simsopt_jax_adapters.geo.flat675.build_flat675_problem` fits any
compatible simsopt boundary onto that layout and refuses, rather than
silently reshapes, a boundary the layout cannot represent.  This example is
built from repository test-file geometry so it runs from a clean clone.

WHERE TO RUN IT, AND WHAT IS ACTUALLY CERTIFIED
-----------------------------------------------
On a GPU against a native CPU denominator this production lane measured
1.67x at equal budget 3, 7.70x at the headline equal budget 37, and
7.36x quality-matched, all on process wall.  That measurement is scoped to
the FROZEN-BUNDLE configuration at the one archived start candidate --
reachable here with ``--bundle`` when that host-local bundle is present.
The configuration this script ships by default runs the same production
lane on repository geometry and makes NO timing claim of its own.

Cold start is disclosed, not claimed: the first solve in a process pays the
full XLA compile of the fused program (~150 s cold, N=1, reported and never
claimed).  The win regime is repeated or warm work in a process that has
already compiled -- which is also why ``--smoke`` here is dominated by
compilation rather than by arithmetic.  That cost is not something this
script can tune away.

``--smoke`` runs the same production lane on a deliberately small problem for
a couple of iterations.  Its ``ok`` status means the fused lane executed and
stayed finite, never that anything converged.

FINAL SURFACE CHECK
-------------------
``--polish`` freezes the coils and vessel, then runs reduced Schur Newton
after the fused optimization.  Its stationarity check uses the nested
contract's 1e-13 physics threshold; this is a numerical condition, not a
complete physical-validity certificate.  Boozer equation RMS, surface-label
error, surface movement, and objective increase require explicit acceptance
limits.  Without them acceptance is not assessed.  Failed corrections are
reported as rejected.  The polish time is separate from the fused solve,
and the historical speedups above do not include it.

Polish is a single final correction, with no automatic retry.  A higher-weight
flat restart would change the objective and could leave stationarity again;
it would need another final check and has no established cost advantage.

WHERE THE CODE IS
-----------------
``simsopt_jax_adapters.examples.single_stage_flat675``.  This tier directory
is not an importable package, so the two configurations, the guarded fused
solve, and the CLI live in that module, where the benchmarks and the contract
tests import them instead of loading this file by location.
"""

from __future__ import annotations

from simsopt_jax_adapters.examples.single_stage_flat675 import main

if __name__ == "__main__":
    raise SystemExit(main(description=__doc__))
