#!/usr/bin/env python3
"""Native C++/simsoptpp twin of the flat coupled single-stage example.

Coil, vessel, and Boozer-surface degrees of freedom are one state vector.
There is no nested equilibrium solve: the rotational transform and the net
poloidal current are the two-column least-squares solution of the Boozer
residual, and the eight production terms are public simsopt objectives on a
native ``BiotSavart``.  The outer loop is host SciPy L-BFGS-B at the archived
genuine-675 policy (``maxcor=300``, ``maxls=8``, ``ftol=0``, ``gtol=1e-3``).

This is the published form of the native lane that the JAX fused example
``examples/jax/3_Advanced/single_stage_flat675.py`` was measured against.
The CLI, budget semantics, and JSON keys match that example.  ``--smoke``
runs the same production objective on a deliberately small quadrature for a
couple of iterations.  ``--bundle`` selects the host-local frozen campaign
input; without it the script is clone-runnable from repository test-file
geometry and makes no timing claim of its own.

The implementation is
``simsopt_jax_adapters.examples.single_stage_flat675_native_twin``: this tier
directory is not an importable package, so the problem construction and the
objective live in that module, where the benchmarks and the contract tests
import them instead of loading this file by location.
"""

from __future__ import annotations

from simsopt_jax_adapters.examples.single_stage_flat675_native_twin import main

if __name__ == "__main__":
    raise SystemExit(main(description=__doc__))
