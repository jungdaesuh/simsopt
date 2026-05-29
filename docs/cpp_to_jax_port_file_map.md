# C++ → JAX Port File Map

> **"Ported from C++"** means a JAX `.py` file that reimplements a `simsoptpp`
> C++ kernel — distinct from the ~40 JAX modules that port *Python* simsopt.

**As of:** 2026-05-29 · **Branch:** `gpu-purity-stage2-20260405`

**Primary source:** the repo's own
`.artifacts/jax_port_gap_audit_2026-05-13/cpp_port_gap.md`
(78 `PORTED` / 23 `PARTIAL` / 7 `UNPORTED` / 11 `NON-PORTABLE` /
8 `UNCLEAR` of ~127 public C++ symbols), lifted from symbol-level to
file-level and re-verified against the current tree. Python-vs-C++ provenance
of the ambiguous field/surface classes was confirmed by source search against
`src/simsoptpp/`.

---

## Legend

`Status` is the single source of truth. It answers one question: **did the
C++ symbol's runtime capability get reimplemented as a differentiable JAX
path?** `PARTIAL` is reserved strictly for a *genuine partial port* — it is
**never** used for code that is intentionally not ported, nor for code that is
not a differentiable kernel. Those are `NON-PORTABLE`.

| Status | Meaning |
|---|---|
| **PORTED** | A current JAX equivalent exists for the C++-grounded runtime surface named in the row. Inline notes may still describe deliberate API-shape differences, autodiff replacements, or unsupported behavior inherited from C++. |
| **PARTIAL** | A JAX module ports the differentiable kernel, but a genuine sub-capability of that kernel is still missing (noted inline). Reserved for **real gaps** only — never for API-shape/cache-orchestration differences, intentional non-ports, or non-differentiable code. |
| **NON-PORTABLE** | The C++ symbol is intentionally not mirrored because it is cache orchestration, pybind11 glue, or object-lifecycle / bootstrap code with no differentiable JAX analogue. The underlying numerical capability, where one exists, is reached through other rows. |
| **UNPORTED** | A specific C++ symbol has no JAX counterpart yet (noted inline). |
| **UNCLEAR** | C++/JAX semantic correspondence is ambiguous (mostly LinAlg helpers / pybind trampolines); see the underlying audit's open questions. |

**C++ file-naming pattern** (why most "oracle" files below are `.h`, not
`.cpp`): many low-level simsopt numerics are C++ *templates*, so substantial
implementation often lives in `*_impl.h` / `*.h`. This is a routing hint, not
a rule: files such as `dommaschk.cpp`, `magneticfield_wireframe.cpp`,
`permanent_magnet_optimization.cpp`, and `tracing.cpp` contain real
implementation code. `*_c.cpp` generally instantiates templates for concrete
types; `*_py.cpp` / `python_*.cpp` are pybind11 bindings; `py*.h` are
trampoline headers.

---

## 1. Biot-Savart & magnetic-field core

| C++ source | JAX port file(s) | Status |
|---|---|---|
| `biot_savart_impl.h`, `biot_savart_c.cpp`, `biot_savart_py.cpp` | `jax_core/biotsavart.py`, `jax_core/biotsavart_cpu_ordered.py`, `field/biotsavart_jax.py` (shim), `jax_core/field.py` (grouped accumulation) | PORTED |
| `biot_savart_vjp_impl.h`, `biot_savart_vjp_c.cpp`, `biot_savart_vjp_py.cpp` | folded into `jax_core/biotsavart.py` (`jax.vjp`) + `field/biotsavart_jax_backend.py` | PORTED |
| `magneticfield_biotsavart.cpp/.h` | `field/biotsavart_jax_backend.py` (`BiotSavartJAX`, `SpecBackedBiotSavartJAX`) | NON-PORTABLE — `B`/`A`/`dB`/`dA`/VJP accessors are JAX-native; only the legacy `compute(derivatives=N)` cache-bundle entrypoint is intentionally not mirrored (audit `cpp_port_gap.md:69` "NON-PORTABLE cache orchestration") |
| `magneticfield.h` (abstract `B`/`dB`/`A`/`AbsB`/`GradAbsB`) | `jax_core/field.py` + per-field backends below | NON-PORTABLE — the abstract mutable-cache base class is replaced by per-field backends and protocols rather than a one-to-one cache API; every concrete field kernel is ported in its own row |
| `magneticfield_interpolated.h` | `jax_core/interpolated_field.py`, `field/interpolated_field_jax.py` | PORTED |
| `magneticfield_wireframe.cpp/.h`, `wireframe_field_impl.h` | `jax_core/wireframe.py`, `field/wireframefield_jax.py` | PORTED — second spatial derivatives are unsupported on both the C++ and JAX wireframe paths |
| `dommaschk.cpp/.h` | `jax_core/analytic_fields.py` (`dommaschk_B/dB`), `field/dommaschk_jax.py` | PORTED |
| `reiman.cpp/.h` | `jax_core/analytic_fields.py` (`reiman_B/dB`), `field/reiman_jax.py` | PORTED |
| `regular_grid_interpolant_3d{.h,_c.cpp,_impl.h,_py.cpp}` | `jax_core/regular_grid_interp.py` | PORTED |
| `coil.h`, `current.h` | `jax_core/specs.py` (`CoilSpec`, `CurrentValueSpec`) | PORTED |

## 2. Boozer magnetic field

| C++ source | JAX port file(s) | Status |
|---|---|---|
| `boozermagneticfield.h` (analytic + ~33 scalar accessors) | `jax_core/boozer_analytic.py`, `jax_core/boozer_fixed_state.py`, `field/boozermagneticfield_jax.py` | PORTED |
| `boozermagneticfield_interpolated.h` | `jax_core/interpolated_boozer_field.py`, `field/boozermagneticfield_jax.py` | PORTED |
| `boozerradialinterpolant.cpp/.h` | `jax_core/boozer_radial_interp.py`, `jax_core/boozer_radial_field.py` | PORTED |

## 3. Surfaces

| C++ source | JAX port file(s) | Status |
|---|---|---|
| `surface.cpp/.h` — evaluation base (area/volume/normal/curvatures/fundamental-forms) | `jax_core/surface_integrals.py` + folded into the per-kind modules below | PORTED |
| `surface.cpp/.h` — construction/bootstrap helpers (`fit_to_curve`, `least_squares_fit`, `extend_via_*`) | none — remain CPU/object lifecycle APIs | NON-PORTABLE — object-mutation/bootstrap workflows, not differentiable hot-path kernels |
| `surfacerzfourier.cpp/.h` | `jax_core/surface_rzfourier.py` | PORTED — 3rd-derivative `_lin` variants are present in `surface_rzfourier.py` |
| `surfacexyzfourier.cpp/.h` | `jax_core/surface_fourier.py` | PORTED — fundamental forms, curvatures, derivative helpers, and 3rd-derivative `_lin` variants are present |
| `surfacexyztensorfourier.h` | `jax_core/surface_fourier.py`, `jax_core/surface_fourier_kernels.py`, `jax_core/surface_fourier_indices.py`, `geo/surface_fourier_jax.py` (shim), `geo/surface_fourier_jax_cpu_ordered.py` | PORTED |

## 4. Curves

| C++ source | JAX port file(s) | Status |
|---|---|---|
| `curve.cpp/.h` (base: kappa/torsion/arclength) | `jax_core/curve_geometry.py`, `jax_core/curve_kernels.py` | PORTED — pure arclength/kappa/torsion helpers are promoted into `jax_core` |
| `curvexyzfourier.cpp/.h` | `jax_core/curve_xyz_fourier.py`, `jax_core/curve_xyz_fourier_symmetries.py` | PORTED |
| `curverzfourier.cpp/.h` | `jax_core/curve_rz_fourier.py` | PORTED |
| `curveplanarfourier.cpp/.h` | `jax_core/curve_planar_fourier.py` | PORTED |
| `python_distance.cpp` (`get_pointclouds_closer_than_threshold*`, `compute_linking_number`) | `geo/_distance_jax.py`, `jax_core/curve_geometry.py` (`pair_linking_number_pure`) | PORTED |

## 5. Boozer residual, objectives, magnets, optimizers, tracing

| C++ source | JAX port file(s) | Status |
|---|---|---|
| `boozerresidual_impl.h`, `boozerresidual_py.cpp/.h` | `geo/boozer_residual_jax.py` | PORTED — direct `boozer_dresidual_dc` kernel replaced by autodiff over the primal |
| `integral_BdotN.cpp/.h` | `jax_core/integral_bdotn.py`, `objectives/integral_bdotn_jax.py` (shim) | PORTED |
| `dipole_field.cpp/.h` | `jax_core/dipole_field.py`, `field/dipole_field_jax.py`, `geo/permanent_magnet_grid_jax.py` | PORTED |
| `permanent_magnet_optimization.cpp/.h` (MwPGP, GPMO\*) | `jax_core/pm_optimization.py`, `jax_core/pm_workflow.py`, `solve/permanent_magnet_optimization_jax.py` | PORTED |
| `wireframe_optimization.cpp/.h` (GSCO) | `jax_core/wireframe_workflow.py`, `solve/wireframe_optimization_jax.py` | PORTED |
| `tracing.cpp/.h` (+ `python_tracing.cpp`) | `jax_core/tracing.py` | PORTED |

---

## Boundary — Python-logic wrappers that *drive* C++ kernels

These files consume the C++-ported kernels above, but their own algorithm
ports upstream **Python** (`boozersurface.py`, `surfaceobjectives.py`,
`fluxobjective.py`), so they are **not** themselves C++ ports:

- `geo/boozersurface_jax.py`
- `geo/optimizer_jax.py`, `geo/optimizer_jax_reference.py`
- `geo/label_constraints_jax.py`
- `geo/surfaceobjectives_jax.py`, `geo/surfaceobjectives_traceable_jax.py`
- `objectives/fluxobjective_jax.py`, `jax_core/objectives_flux.py`
- `geo/curveobjectives_jax.py`

## Not ported from C++ (port Python simsopt — no `simsoptpp` kernel exists)

Verified Python-only (no corresponding C++ kernel in `src/simsoptpp/`):

- **Analytic / extra fields:** `jax_core/circular_coil.py`, `field/circular_coil_jax.py`, `field/toroidal_field_jax.py`, `field/poloidal_field_jax.py`, `field/mirror_model_jax.py`, `field/magneticfieldclasses_jax.py`, `field/scalar_potential_rz_jax.py`, `jax_core/scalar_potential_rz.py`, `jax_core/magneticfield_composition.py`, `jax_core/analytic_pure_fields.py`
- **Surfaces / curves (Python representations):** `jax_core/surface_henneberg.py`, `jax_core/curve_helical.py`, `jax_core/oriented_curve.py`, `jax_core/framedcurve.py`, `jax_core/finitebuild.py`. Wrappers are not one-to-one: this subset has `geo/framedcurve_jax.py`, while other helpers are consumed through Python host modules/spec methods rather than same-named `geo/*_jax.py` files.
- **QFM:** `jax_core/qfm_solver.py`, `geo/qfmsurface_jax.py`
- **MHD / VMEC:** `jax_core/profiles.py`, `mhd/profiles_jax.py`, `jax_core/redl_current.py`, `jax_core/mhd_bootstrap.py`, `mhd/bootstrap_jax.py`, `jax_core/mhd_reductions.py`, `jax_core/vmec_geometry.py`, `jax_core/vmec_fieldlines.py`, `mhd/vmec_diagnostics_jax.py`
- **Stage 2 / objectives:** `objectives/stage2_target_objective_jax.py`
- **Solve wrappers:** `solve/serial_jax.py`, `solve/mpi_jax.py`
- **Misc:** `jax_core/magnetic_axis_helpers.py`, `jax_core/sampling.py`, `field/sampling_jax.py`, `jax_core/surface_classifier.py`
- **Runtime/JAX infrastructure with no direct `simsoptpp` kernel counterpart:** `backend/*`, `jax_core/sharding.py`, `jax_core/reductions.py`, `jax_core/_math_utils.py`, `jax_core/_vector_norms.py`, `jax_core/_elliptic.py`, `jax_core/_spline_utils.py`, `jax_core/_root.py`, `jax_core/_finite_difference.py`, `jax_core/_sympy_to_jax.py`, `jax_core/_device_scalars.py`. `jax_core/specs.py` is listed in the C++-port table above for `CoilSpec`/`CurrentValueSpec`; the same file also contains Python/JAX-only specs.

---

## Provenance & caveats

- **Generated:** 2026-05-29, from `.artifacts/jax_port_gap_audit_2026-05-13/cpp_port_gap.md` plus live source/test-file checks.
- **File-level mapping:** current against the `gpu-purity-stage2-20260405` tree.
- **Symbol-level line numbers:** the underlying 2026-05-13 audit's *line
  numbers* have partially drifted; the file-level mapping above is current,
  but regenerate the symbol-level audit if you need exact `file:line`
  references.
- This map covers **what** is ported, not a correctness verdict. For the
  latter, use `docs/jax_parity_manifest.md`, focused parity tests, and the
  subsystem parity artifacts under `.artifacts/parity_audit_2026-05-16/`.
