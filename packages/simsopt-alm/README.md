# simsopt-alm

An augmented Lagrangian (ALM) solver for

    minimize f(x)  subject to  g_i(x) <= 0,

written for simsopt's Stage-2 and single-stage coil optimization, and usable
for any problem whose value and gradient you can compute. It is a pure-Python
package (import name `simsopt_alm`) that installs beside the simsopt you
already have, whether a fork, an editable checkout or a PyPI release, and
changes nothing in it.

- The solver (`simsopt_alm` and its modules) needs only Python >= 3.8, numpy
  and scipy. Importing it loads no simsopt module.
- `simsopt_alm.signed_constraints` holds smooth signed coil rows (minimum
  coil-coil and coil-surface distance, maximum curvature). It imports
  simsopt's `Derivative`, so it needs a simsopt in the same environment.

## Install

The package lives in `packages/simsopt-alm/` of the `alm-library` branch of
<https://github.com/jungdaesuh/simsopt>. It is not on PyPI.

Plain install, next to your own simsopt:

```sh
pip install "git+https://github.com/jungdaesuh/simsopt.git@alm-library#subdirectory=packages/simsopt-alm"
```

Editable install, to read or change the solver's sources:

```sh
git clone -b alm-library https://github.com/jungdaesuh/simsopt.git simsopt-alm-clone
pip install -e simsopt-alm-clone/packages/simsopt-alm
```

Without a simsopt of your own, the `simsopt` extra installs PyPI's:
`pip install "simsopt-alm[simsopt] @ git+https://github.com/jungdaesuh/simsopt.git@alm-library#subdirectory=packages/simsopt-alm"`.

## Quick start

```python
import numpy as np
from simsopt_alm import ALMPhysics, ALMSettings, cached_alm_evaluator, minimize_alm


def physics(x):
    # f = ||x||^2 and one row g = 1 - x0 <= 0; the solution is x = (1, 0).
    return ALMPhysics(
        base_value=float(x @ x),
        base_grad=2.0 * x,
        constraint_values=np.array([1.0 - x[0]]),
        constraint_grads=(np.array([-1.0, 0.0]),),
    )


result = minimize_alm(np.array([3.0, 2.0]), ["x0_at_least_one"],
                      cached_alm_evaluator(physics), ALMSettings(), {"maxiter": 200})
print(result.termination_reason, result.x.round(4) + 0.0)  # converged [1. 0.]
```

For simsopt objectives, `alm_problem_physics(dofs, objective, inequalities)`
builds the `ALMPhysics` from an `Optimizable` and a list of rows, each a
function of that objective: the kernels of `simsopt_alm.signed_constraints`
with their leading arguments bound, or `partial(signed_upper_bound, objective,
bound)` for any simsopt objective. The examples in
`examples/` show complete problems:

- `stage_two_optimization_alm.py`: Stage-2 coils for the QA target of
  arXiv:2108.03711 (shipped as `simsopt_alm/data/input.LandremanPaul2021_QA`,
  simsopt's own test file), with the distance and curvature terms as rows;
- `boozerQA_alm.py`: single-stage quasi-axisymmetry of the NCSX coils on a
  Boozer surface re-solved at every evaluation, a stateful evaluator;
- `alm_composition_example.py`: history, checkpoints and resume;
- `alm_typed_evaluator_example.py`: an evaluator typed against
  `ALMEvaluation`.

`help(simsopt_alm)` documents the evaluator contract, hybrid (smooth and
exact) rows, stateful evaluators, history and checkpoints.

## What `converged` means

`result.success` is true, with `termination_reason` `converged` (or
`constraints_inactive_converged` for a hybrid problem whose exact rows are
all strictly inactive), when the returned point `result.x` is an approximate
KKT point at the shifted multipliers `λ⁺ = max(0, λ + ρ g)`:

- feasibility: the maximum violation is at most `feasibility_tol` (for a
  hybrid problem, in both the smooth and the exact rows);
- stationarity: the norm of the augmented Lagrangian's gradient, without the
  components that point out of the box at an active `base_bounds` bound, is
  at most `stationarity_tol`;
- complementarity: the gap `Σ_i λ⁺_i max(0, -g_i)` is at most
  `feasibility_tol` (absolute, in the units of f: scale f to O(1));
- and no hybrid signal mismatch or binding multiplier cap.

This certifies the returned point only. It is not global optimality, not a
guarantee of descent from the start, and not a statement about which basin
a nonconvex problem ends in. `policy.py` (`_kkt_point`) makes the decision and
`core.py` (`_complementarity_gap`) measures the gap. Every other termination
reason is a stop without that certificate; `result.restored_best_feasible`
says when the solver returned the best feasible iterate it saw instead of
the last one.

## Setup skill and guide

The `alm-library` branch also ships a Claude Code skill,
`.claude/skills/simsopt-alm-setup/`, that installs this package, interviews
you about the problem, generates a problem module and a runner from tested
templates (Stage 2, Boozer single-stage, generic), checks gradients and
constraint signs, and explains every termination reason. Its content, for
reading by hand, is `docs/alm_setup_guide.md` on the same branch.

## Tests

From this directory, with the package and simsopt installed:

```sh
python -m pytest tests
```

The golden trajectories in `tests/alm_golden/` replay bit for bit only in
their recording environment (`manifest.json`: numpy, SciPy, x86_64 and
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OPENBLAS_CORETYPE=Haswell`);
elsewhere they replay within tolerances measured from last-bit noise.
