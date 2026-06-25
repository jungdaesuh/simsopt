# JAX Port — Parity Status

Branch `gpu-purity-stage2-20260405`. CPU/C++ oracle = installed `simsoptpp` + upstream Python; FD; closed form; pinned datasets.
Tolerance SSOT: `benchmarks/validation_ladder_contract.py::PARITY_LADDER_TOLERANCES`.

## 1. Same-candidate kernel parity (oracle-anchored)

| Subsystem | Oracle | Agreement | Source |
|---|---|---|---|
| Biot–Savart `B`, `dB/dX` | simsoptpp | 1e-12–1e-15 | review 2026-05-28 |
| Surface eval (XYZTensor/RZFourier) | simsoptpp | 1e-10 | review 2026-05-28 |
| Surface DOF Jacobians | simsoptpp + FD | rtol 1e-8 / atol 1e-10 | derivative-heavy lane |
| Boozer residual (scalar) | simsoptpp | 1e-10 | direct-kernel lane |
| Boozer grad / Jacobian (composed) | FD + simsoptpp | rtol 1e-8 / atol 1e-10 | derivative-heavy lane |
| Boozer penalty Hessian (column-complete sweep) | simsoptpp | rtol 1e-8 / atol 1e-10 | direct-hessian-oracle lane |
| `integral_BdotN` (3 defs) | sopp | 1e-10 | direct-kernel lane |
| CircularCoil `B`, `A`, `dB/dX` + toroidal arrangement | upstream Python + legacy reference | direct-kernel lane / legacy magnitude tolerance | `tests/field/test_circular_coil_jax.py` |
| MagneticFieldSum/Multiply public wrappers | upstream Python | direct-kernel lane | `tests/field/test_magnetic_field_composition_jax.py` |
| InterpolatedField `B`, `GradAbsB`, folds, cylindrical projections, convergence | upstream Python + analytic toroidal source | direct-kernel lane / refinement contract | `tests/field/test_interpolated_field_jax_item15.py` |
| Wireframe fields, constraints, matrices, collision constraints, windowpane state, optimizer errors/targets | simsoptpp + upstream Python | direct-kernel lane / public constraint and optimizer contract | `tests/jax/core/test_wireframe_item29.py`; `tests/field/test_wireframefield_jax_item30.py`; `tests/solve/test_wireframe_optimization_jax_item31.py` |
| Particle tracing GC/full proximity, Cartesian energy/moment/angular-momentum invariants, Boozer canonical momentum, and Boozer transit/flux stopping diagnostics | legacy particle invariant formulas | public Cartesian wrapper contract plus core Boozer guiding-centre contract | `tests/field/test_particle_jax_invariants.py`; `tests/jax/core/test_tracing_jax_conservation.py` |
| Curve geometry, coefficient derivatives, curvature/torsion/kappadash derivatives, Frenet-frame derivatives, centroid | upstream Python/C++ curve methods | direct-kernel lane / tolerance-based derivative parity | `tests/geo/test_curve_objectives_jax.py`; `tests/geo/test_curvexyzfouriersymmetries_spec_jax.py`; `tests/geo/test_framedcurve_jax_item18.py` |
| Surface aspect-ratio, scalar metrics, non-RZ coordinate derivatives, area/volume Hessians | upstream Python/C++ surface methods | derivative-heavy lane / tolerance-based parity | `tests/geo/test_surface_objectives_jax.py`; `tests/geo/test_surface_fourier_jax.py`; `tests/geo/test_surface_rzfourier_jax.py` |
| QFM objective, label constraint, and penalty gradients | upstream `QfmSurface` public methods | tolerance-based CPU/JAX wrapper parity | `tests/geo/test_qfmsurface_jax.py` |
| Framed-curve strain penalties for torsion and binormal curvature | public strain penalty finite-difference contract | directional-FD contract parity | `tests/geo/test_framedcurve_jax_wrappers_item18.py` |
| stellsym DOF scatter | simsoptpp DOF probe | 0 / 52 mismatches | review 2026-05-28 |
| LS `run_code` iota | upstream | 3.9e-16 | review 2026-05-28 |
| LS `run_code` 181-DOF state | upstream | 1.1e-15 | review 2026-05-28 |
| `compute_G_from_currents` | upstream | byte-identical | review 2026-05-28 |
| Single-stage IFT grad (LS path) | scipy LS | rtol 1e-10 | 2026-04-06 |
| Stage-2 `SquaredFlux` value+grad | simsoptpp | machine precision | M2 |

Review 2026-05-28: 25 candidate discrepancies, 0 real bugs, span 1e-12…1e-18.

## 2. Parity ladder lanes

| Lane | rtol | atol | residual | Scope |
|---|---|---|---|---|
| direct-kernel | 1e-10 | — | — | same-state B, surface γ, integral_BdotN, raw Boozer residual, LS-wrapper grad |
| derivative-heavy | 1e-8 | 1e-10 | — | dB/dX, BS VJP, surface coeff Jacobians, composed Boozer Jacobian |
| direct-hessian-oracle | 1e-8 | 1e-10 | — | Boozer penalty Hessian basis sweep |
| exact_well_conditioned_adjoint | 1e-6 | 1e-8 | ≤1e-10 | operator-vs-PLU adjoint vector parity, IotasJAX grad |
| exact_ill_conditioned_adjoint | — | — | residual/failure-only | no vector parity asserted |
| state-parity gate | 1e-12 (same-machine) | — | sdofs_inf ≤ 1e-11 | byte-identity (cross-machine ≥3× worst) |

float32 smoke floor = sqrt(eps_f32); float64 floor = 1e-14, cap = 1e-10.
Modes: 6 (`native_cpu`, `jax_cpu_{fast,parity,float32_smoke}`, `jax_gpu_{fast,parity}`). `*_parity` → byte-identity; `*_fast` → no byte-identity claim.

## 3. End-to-end real-run ladder (CPU-twin vs JAX)

`benchmarks/single_stage_init_parity.py` — full `run_code()` on VMEC-seeded equilibria; tiered + `full_run_artifact_contract`.

| Tier | Number | Status | Source |
|---|---|---|---|
| same_candidate_replay (obj/grad/boozer abs-diff) | 0.0 | ✅ | G3 strict ladder, this session |
| init-parity census: iota | 4.5e-17 | ✅ | GCE 2026-03-21 tier3 |
| init-parity census: volume | 9.8e-16 | ✅ | GCE 2026-03-21 tier3 |
| init-parity census: field-error | 9.1e-16 | ✅ | GCE 2026-03-21 tier3 |
| objective value rel-diff | 3.4e-7 | ✅ | GCE 2026-03-21 tier2 |
| field-error rel-diff | 4.5e-8 | ✅ | GCE 2026-03-21 tier2 |
| geometry rel-diff | 1.7e-6 | ✅ | GCE 2026-03-21 tier2 |
| adjoint-FD residual | 1.2e-15 | ✅ | GCE 2026-03-21 tier4 |
| adjoint-FD max sample rel-err | 3.2e-4 | ✅ | GCE 2026-03-21 tier4 |
| at-optimum iota25: obj \|Δ\| / iota \|Δ\| | 5.3e-8 / 4.6e-5 | ✅ | seed replay, this session |
| at-optimum iota0064: obj \|Δ\| / iota \|Δ\| | 4e-19 / 2.3e-17 | ✅ | seed replay, this session |
| strict end-state byte-identity @ production maxiter | — | ⏳ open | not collected |

## 4. Device parity (JAX-CPU vs JAX-CUDA)

| Metric | Number | Source |
|---|---|---|
| cold-start Newton iota, CPU | 0.1477642472606855 | C1, this session |
| cold-start Newton iota, GPU | 0.1477642472599228 | G1, this session |
| CPU↔GPU iota \|Δ\| | ~7e-13 | derived |
| compile-once (m04 / m10), both devices | 177 / 179 | this session |
| Newton ‖grad‖ at convergence | 8.1e-12 | G1/C1 |

## 5. GPU parity

| Metric | Number | Source |
|---|---|---|
| L4 test pass | 254/255 (stale) → 275/275 clean | commit 07c568e1 |
| V100 ULP | 2 | 2026-04-02 |
| V100 reproducibility | bitwise | 2026-04-02 |
| V100 VRAM | 238 MB | 2026-04-02 |
| V100 grouped-adjoint speedup | ~33× | 2026-04-02 |
| A100 single-stage (lm-minpack) | 0 transfer crashes; gate-blocked, now fixed | `jax_transfer_guard_validation_2026-05-29.md` |
| A100 lm-minpack solver / residency contract | validated (full results) | `jax_transfer_guard_validation_2026-05-29.md` |

## 6. Known numeric limits

| Item | Number |
|---|---|
| exact-path adjoint: CPU vs JAX vector norm ratio | ~3× |
| exact-path adjoint: `Jᵀadj=rhs` residual (both) | 1e-15 |
| cross-machine LS state `sdofs_inf` (2 macOS hosts) | 1.9e-14 – 3.6e-12 |
| cross-machine κ(H) | 5.3e4 (hardware-invariant) |
| cross-machine `H_inf_diff` | 1.4e-12 – 9.6e-10 |

Exact-path adjoint vector parity is FD-validated per path, not cross-compared (ill-conditioned Newton Jacobian).

## 7. Reproduce

```bash
export PYTHONNOUSERSITE=1 PYTHONPATH=src JAX_ENABLE_X64=True JAX_PLATFORM_NAME=cpu

# pure-JAX kernel + parity (no simsoptpp)
pytest tests/field/test_biotsavart_jax.py tests/geo/test_surface_fourier_jax.py \
       tests/geo/test_boozer_residual_jax.py tests/geo/test_boozer_derivatives_jax.py \
       tests/objectives/test_integral_bdotn_jax.py

# Boozer solver + single-stage IFT
pytest tests/geo/test_boozersurface_jax.py tests/geo/test_surface_objectives_jax.py \
       tests/integration/test_single_stage_jax.py

# integration parity (needs simsoptpp)
pytest tests/integration/

# end-to-end real-run ladder
python benchmarks/single_stage_init_parity.py
```

## 8. Reproduced counts (this session, consolidated tree)

| Suite | Result |
|---|---|
| boozersurface + biotsavart + surface_fourier + boozer_derivatives | 702 passed, 4 skipped |
| transfer-guard + CPU-geo-import + curve regression set | 38 passed |
| optimizer resolver (item 13) behavioral snapshot | 940/940 byte-identical |
