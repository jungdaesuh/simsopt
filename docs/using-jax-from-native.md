# Using JAX with existing native simsopt code

The fork branch `jax-port-all` adds JAX versions of simsopt's stage-two field and objective classes and of the
Boozer surface solver. They work on the native coils, curves, currents and surfaces you already build, so a native
script changes in a few lines. With the fused wrapper, a stage-two objective and its gradient run as one compiled
program, on CPU or GPU.

## Install

You need Python 3.12 or newer.

```
git clone -b jax-port-all https://github.com/jungdaesuh/simsopt.git
cd simsopt
pip install -e ".[jax]"
pip install "jax[cuda12]==0.12.0.1"    # NVIDIA GPU only
```

Tested with Python 3.12.13, JAX 0.12.0.1 (on CPU, and on an A100 with the CUDA 12 plugin), SciPy 1.18.1 and NumPy
2.5.4.

## What stays the same

- Build coils, curves, currents and surfaces with the native classes, as before. The JAX classes read these objects
  and their DOFs directly; there is no conversion step.
- `JF.x`, `JF.J()`, `JF.dJ()`, `fix_all()`, `Weight`, `QuadraticPenalty`, `+` and `*` work as before. The objective
  has the same DOFs, with the same names and order, as the native one.
- Your optimizer loop does not change.
- Native objectives built on a Boozer surface (`NonQuasiSymmetricRatio`, `Iotas`, `MajorRadius`) take a
  `JaxBoozerSurface` as they take a `BoozerSurface`.

## What to change

1. Call `set_backend("jax", device="gpu")` (or `device="cpu"`) at the top of the script, before anything uses JAX
   and before you build the field. It is in `simsopt_jax.backend`.
2. Replace each native class with its JAX class from the table. Each JAX class takes the same constructor
   arguments as its native class: same names, order and defaults.
3. For a stage-two objective, wrap the finished sum once with `JaxFusedStageTwoObjective` and give the wrapper to
   the optimizer.

| native class | JAX class | import the JAX class from | differences in use |
|---|---|---|---|
| `simsopt.field.B2Energy` | `JaxB2Energy` | `simsopt_jax_adapters.field` | none |
| `simsopt.field.BiotSavart` | `JaxBiotSavart` | `simsopt_jax_adapters.field` | Field values come back as JAX arrays. Not a native `MagneticField`: no field-line tracing, `InterpolatedField` or field arithmetic. No `clear_cached_properties`, `compute`, `invalidate_cache`, `to_mgrid`, `to_vtk`, `*_ref`, `fieldcache_*`. |
| `simsopt.field.LpCurveForce` | `JaxLpCurveForce` | `simsopt_jax_adapters.field` | none |
| `simsopt.field.LpCurveTorque` | `JaxLpCurveTorque` | `simsopt_jax_adapters.field` | none |
| `simsopt.field.NetFluxes` | `JaxNetFluxes` | `simsopt_jax_adapters.field` | Not taken by the fused wrapper. |
| `simsopt.field.SquaredMeanForce` | `JaxSquaredMeanForce` | `simsopt_jax_adapters.field` | none |
| `simsopt.field.SquaredMeanTorque` | `JaxSquaredMeanTorque` | `simsopt_jax_adapters.field` | none |
| `simsopt.objectives.SquaredFlux` | `JaxSquaredFlux` | `simsopt_jax_adapters.objectives` | Needs a `JaxBiotSavart`. Reads the surface once: changing the surface DOFs later raises a RuntimeError. |
| `simsopt.geo.CurveCurveDistance` | `JaxCurveCurveDistance` | `simsopt_jax_adapters.geo` | none |
| `simsopt.geo.CurveLength` | `JaxCurveLength` | `simsopt_jax_adapters.geo` | none |
| `simsopt.geo.CurveSurfaceDistance` | `JaxCurveSurfaceDistance` | `simsopt_jax_adapters.geo` | none |
| `simsopt.geo.LpCurveCurvature` | `JaxLpCurveCurvature` | `simsopt_jax_adapters.geo` | none |
| `simsopt.geo.MeanSquaredCurvature` | `JaxMeanSquaredCurvature` | `simsopt_jax_adapters.geo` | Inside the fused wrapper, only inside `QuadraticPenalty`. |
| `simsopt.geo.BoozerResidual` | `JaxBoozerResidual` | `simsopt_jax_adapters.geo.boozer_surface` | Needs a `JaxBoozerSurface` and a `JaxBiotSavart`. Not taken by the fused wrapper. No `dJ_by_dB`. |
| `simsopt.geo.BoozerSurface` | `JaxBoozerSurface` | `simsopt_jax_adapters.geo.boozer_surface` | Needs a `JaxBiotSavart`. Not taken by the fused wrapper. No `boozer_exact_constraints`, `boozer_penalty_constraints_vectorized`, `minimize_boozer_exact_constraints_newton`. |
| (none) | `JaxFusedStageTwoObjective` | `simsopt_jax_adapters.objectives` | wraps the finished objective |

## Before and after

Setup, the same for both:

```python
import numpy as np
from scipy.optimize import minimize
from simsopt.field import Current, coils_via_symmetries
from simsopt.geo import SurfaceRZFourier, create_equally_spaced_curves
from simsopt.objectives import QuadraticPenalty, Weight

ncoils = 4
s = SurfaceRZFourier.from_vmec_input("tests/test_files/input.LandremanPaul2021_QA",
                                     range="half period", nphi=32, ntheta=32)
base_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=1.0, R1=0.5, order=5)
base_currents = [Current(1e5) for _ in range(ncoils)]
base_currents[0].fix_all()
coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
curves = [c.curve for c in coils]
LENGTH_WEIGHT = Weight(1e-6)
```

Native objective:

```python
from simsopt.field import BiotSavart
from simsopt.geo import CurveCurveDistance, CurveLength, LpCurveCurvature, MeanSquaredCurvature
from simsopt.objectives import SquaredFlux

bs = BiotSavart(coils)
JF = (SquaredFlux(s, bs)
      + LENGTH_WEIGHT * sum(CurveLength(c) for c in base_curves)
      + 1000 * CurveCurveDistance(curves, 0.1, num_basecurves=ncoils)
      + 1e-6 * sum(LpCurveCurvature(c, 2, 5.0) for c in base_curves)
      + 1e-6 * sum(QuadraticPenalty(MeanSquaredCurvature(c), 5.0, "max") for c in base_curves))
```

JAX objective:

```python
from simsopt_jax.backend import set_backend
from simsopt_jax_adapters.field import JaxBiotSavart
from simsopt_jax_adapters.geo import (JaxCurveCurveDistance, JaxCurveLength, JaxLpCurveCurvature,
                                      JaxMeanSquaredCurvature)
from simsopt_jax_adapters.objectives import JaxFusedStageTwoObjective, JaxSquaredFlux

set_backend("jax", device="cpu")    # or device="gpu"; call it before building the field
bs = JaxBiotSavart(coils)
JF = (JaxSquaredFlux(s, bs)
      + LENGTH_WEIGHT * sum(JaxCurveLength(c) for c in base_curves)
      + 1000 * JaxCurveCurveDistance(curves, 0.1, num_basecurves=ncoils)
      + 1e-6 * sum(JaxLpCurveCurvature(c, 2, 5.0) for c in base_curves)
      + 1e-6 * sum(QuadraticPenalty(JaxMeanSquaredCurvature(c), 5.0, "max") for c in base_curves))
JF = JaxFusedStageTwoObjective(JF)  # wrap once, after every term is in JF
```

Optimization, the same for both:

```python
def fun(x):
    JF.x = x
    return JF.J(), JF.dJ()


for _ in range(2):
    res = minimize(fun, JF.x, jac=True, method="L-BFGS-B", options={"maxiter": 100, "maxcor": 300}, tol=1e-15)
    JF.x = res.x
    print(f"{res.nfev} evaluations, J = {res.fun:.3e}")
    LENGTH_WEIGHT *= 0.1    # the next run uses a smaller length weight; no rebuild needed
```

Run it from the top of the clone (the surface file is in `tests/test_files`). The JAX run can take a different
number of evaluations and end with different last digits of J: round-off differs, so L-BFGS-B takes a slightly
different path.

## Rules and limits

The fused wrapper:

- Takes one unscaled `JaxSquaredFlux` and any of `JaxCurveLength`, `JaxLpCurveCurvature` (p=2 only),
  `JaxCurveCurveDistance` (downsample=1 only), `JaxCurveSurfaceDistance`, `JaxLpCurveForce`, `JaxLpCurveTorque`,
  `JaxSquaredMeanForce`, `JaxSquaredMeanTorque`, `JaxB2Energy`, and `QuadraticPenalty` of `JaxCurveLength` or of
  `JaxMeanSquaredCurvature` with `"max"` or `"identity"`. Weights can be numbers or `Weight` objects.
- Per-curve terms of one kind need one term per base coil and share one weight and threshold. Force and torque
  terms target the base coils, with the other coils as coarse sources; `JaxB2Energy` takes all coils.
- Finite-build filament packs work, except with the force, torque and energy terms.
- A term with no fused form, such as a native `CurveLength` or `JaxNetFluxes`, raises a TypeError when you build
  the wrapper. A supported term in a form the program cannot represent raises a ValueError: when you build the
  wrapper, or at the next evaluation for a setting you change later (for example a penalty's `f`).
- The wrapper has to be the whole objective. `wrapper + term`, `2 * wrapper` and `QuadraticPenalty(wrapper, ...)`
  raise a TypeError when the gradient is taken. Put the term into the sum before wrapping.
- Weights, thresholds and penalty targets are read at every evaluation, so you can change them between
  optimizations without a new wrapper.
- `JaxNetFluxes`, `JaxBoozerSurface` and `JaxBoozerResidual` are not fused. Use them without the wrapper; each term
  then runs its own compiled program. Without the wrapper, JAX and native terms can be added in one objective.

The JAX classes:

- `JaxSquaredFlux` reads the surface once, when you build it. If you change the surface, build a new one.
- `JaxBiotSavart` returns JAX arrays: use `np.asarray(bs.B())` where NumPy code needs an array. For field-line
  tracing, `InterpolatedField`, field arithmetic, `to_vtk` or `to_mgrid`, build a native `BiotSavart` on the same
  coils; it shares their DOFs.
- `JaxBiotSavart`, `JaxFusedStageTwoObjective` and `JaxBoozerSurface` hold mutable caches: use each instance in one
  thread only.
- Importing `simsopt_jax` turns on JAX float64 for the whole process.

## Check that the results match

Build both objectives on the same coils: run the native block and keep `JF_native = JF`, then the JAX block and
keep `JF_jax = JF`. Then compare them at the starting point and at a perturbed point:

```python
rng = np.random.default_rng(0)
x0 = JF_native.x
assert JF_native.dof_names == JF_jax.dof_names
for x in (x0, x0 + 1e-2 * rng.standard_normal(x0.size)):
    JF_native.x = x    # the objectives share the coils, so this also moves JF_jax
    J_n, J_j = JF_native.J(), JF_jax.J()
    g_n, g_j = JF_native.dJ(), JF_jax.dJ()
    print(f"J: {J_n:.6e} vs {J_j:.6e}, relative difference {abs(J_j - J_n) / abs(J_n):.1e}; "
          f"dJ relative difference {np.linalg.norm(g_j - g_n) / np.linalg.norm(g_n):.1e}")
```

Both relative differences should be at round-off level.

## Speed

- The first evaluation compiles the program. Later evaluations reuse it, and `J()` and `dJ()` at the same `x` run
  it once.
- New weights, thresholds or fixed DOF values reuse the program. New penalty modes or downsampling, terms added or
  removed, or a different number of free DOFs compile it again.
- Every new process compiles again unless you keep the compiled programs on disk:

  ```
  export JAX_COMPILATION_CACHE_DIR=$HOME/.cache/jax
  export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
  ```

  Without the second line, JAX stores only programs that take a second or more to compile.
- For a GPU, install the CUDA plugin and use `set_backend("jax", device="gpu")`. Called before JAX starts,
  `set_backend` also sets XLA flags that make GPU compiles repeatable (flags already in `XLA_FLAGS` take
  precedence); called later, it warns.

## Examples to copy

These scripts run natively by default and with the JAX classes under `--use-jax`. `--device` is `cpu` (default) or
`gpu`; `--unfused` evaluates each term on its own instead of through the fused wrapper and needs `--use-jax`.

- `examples/2_Intermediate/stage_two_optimization.py`: `--use-jax`, `--device`, `--unfused`
- `examples/3_Advanced/coil_forces.py`: `--use-jax`, `--device`, `--unfused`
- `examples/3_Advanced/stage_two_optimization_finitebuild.py`: `--use-jax`, `--device`, `--unfused`
- `examples/2_Intermediate/boozerQA.py`: `--use-jax`, `--device`

For example: `python examples/2_Intermediate/stage_two_optimization.py --use-jax --device gpu`.
`boozerQA.py --use-jax` uses `JaxBoozerSurface`.

## What to expect

Whole script on one A100 node, measured on the PR branches rather than on this one (stage two on PR 671, coil
forces and finite build on PR 672, boozerQA on PR 675). Native: median of three runs at 8 and at 64 threads. JAX:
the first run, which compiles every program, and in parentheses the median of three runs that load every program
from the disk cache. Evaluation counts differ between runs because round-off changes the optimizer's path.

| command | wall time | evaluations | final J |
|---|---:|---:|---:|
| `stage_two_optimization.py`, native at 8 threads | 27.0 s | 1,152 to 1,182 | 2.0e-06 |
| `stage_two_optimization.py`, native at 64 threads | 62.2 s | 1,080 to 1,152 | 2.0e-06 |
| `stage_two_optimization.py --use-jax --device gpu` | 11.3 s (9.0 s cached) | 1,224 | 2.0e-06 |
| `stage_two_optimization.py --use-jax --unfused --device gpu` | 48.5 s (44.1 s cached) | 1,217 | 2.0e-06 |

| command | wall time | evaluations | final J |
|---|---:|---:|---:|
| `coil_forces.py`, native at 8 threads | 170.8 s | 1,753 to 2,004 | 2.9e-05 |
| `coil_forces.py`, native at 64 threads | 313.4 s | 2,169 to 2,668 | 2.9e-05 |
| `coil_forces.py --use-jax --device gpu` | 18.4 s (14.1 s cached) | 2,023 | 2.9e-05 |
| `coil_forces.py --use-jax --unfused --device gpu` | 85.8 s (77.2 s cached) | 2,048 | 2.9e-05 |
| `stage_two_optimization_finitebuild.py`, native at 8 threads | 119.7 s | 453 to 462 | 5.4e-11 to 5.5e-11 |
| `stage_two_optimization_finitebuild.py`, native at 64 threads | 137.7 s | 455 to 467 | 5.4e-11 to 5.5e-11 |
| `stage_two_optimization_finitebuild.py --use-jax --device gpu` | 22.5 s (12.5 s cached) | 463 | 5.5e-11 |
| `stage_two_optimization_finitebuild.py --use-jax --unfused --device gpu` | 36.9 s (19.3 s cached) | 459 | 5.4e-11 |

| command | wall time | evaluations | final J |
|---|---:|---:|---:|
| `boozerQA.py`, native at 8 threads | 107.7 s | 1,259 | 4.5e-08 |
| `boozerQA.py`, native at 64 threads | 101.0 s | 1,259 | 4.5e-08 |
| `boozerQA.py --use-jax --device gpu` | 144.2 s (109.8 s cached) | 1,248 | 4.3e-08 |
