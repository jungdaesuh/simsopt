# Diagnostic — un-nest vs nested Boozer Newton (LS and exact)

> **Status: diagnostic, not certifying, not a GPU-speed claim.** Analysis-only
> reconstructs on frozen coils. No product test, no `src/` change, no campaign
> seal. **Does not supersede** `docs/receipts/flat675_fused_campaign.md` or
> `docs/receipts/genuine675_fair_bar.md`. Those receipts own F3 / fair-bar
> *speed* on the **flat-675 LS** problem. This note asks a different question:
> does a flat-675 point sit on the nested Boozer manifold?

- **Date:** 2026-08-19 → 2026-08-20
- **Question asked:** can un-nest be checked against C++ Boozer Newton; then
  run it (analysis only); then try both LS and exact purposes without matching
  nested *steps*; then whether public GitHub has a GPU exact solution.
- **Trees used at reconstruct time (read, not modified by this note):**
  - F3 fused endpoint production tree: `simsopt-pr-jax-port-squashed`
    `580217e0cadc427f93721f12aa0f4d8e8915b4e9` (from the B37 pair-set
    `manifest.json` `git.production.commit`; dirty_file_count 0)
  - Native judge / instrument: `simsopt-genuine675-fairbar`
    `1c23f6c5f8964c74cc60f63d81b7f93f2db852f3`
- **Python:** v0c runtime
  `/home/jungdaesuh/simsopt_mixed_artifacts/v0c_62a262b09c_20260715T2150Z/runtime-env/bin/python`
  (3.11, JAX 0.10.0, `JAX_PLATFORMS=cpu`). `simsoptpp` from that env.
  Drivers do `sys.path.insert(0, fairbar)` then `sys.path.insert(0, fairbar/src)`
  and `bootstrap_local_simsopt()` — they do not set `PYTHONPATH`.
  JAX is used only to materialize the native Boozer graph; the Newton judges
  are C++ `BoozerSurface` / `boozer_surface_residual`.
- **Threads:** drivers `os.environ.setdefault("OMP_NUM_THREADS", "16")`; the
  archive does not record the resolved `omp_get_max_threads`. `JAX_ENABLE_X64=1`
- **Inputs:**
  - Frozen bundle `~/simsopt_mixed_artifacts/genuine675-r3-input-1c23f6c5-20260721-r1`
  - Archived native `y_certificate` (GATE 1 / start QR):
    `~/simsopt_mixed_artifacts/genuine675-fixed-budget-maxiter3-r3-1c23f6c5-20260721T124425Z/triad/native_cpp_cpu/lane.json`
    → `result.initial_certificate.y_certificate` (not the final certificate)
  - F3 B37 fused endpoint
    `~/simsopt_mixed_artifacts/flat675_fused_campaign/20260819T163816Z-pairs-b37-2751085/pair2-l1/lane.json`
    (`nit=37`, `nfev=48`, `lane=fused_gpu`)
- **Archived drivers and JSON:**
  `docs/receipts/evidence/boozer_unnest_newton_reconstruct_20260820/`
  (`newton_reconstruct_flat675.{py,json}`, `boozer_ls_exact_purpose.{py,json}`).
  Purpose JSON is the Aug 19 run output (sha256 `2fbd444d…`), including the 661
  LS-projected Fourier dofs at the F3 point (`ι = 0.1408571095530796`).
  Reconstruct JSON is byte-identical to that run (sha256 `1b68f7f6…`). Host-local
  copies of the same bytes:
  `~/simsopt_mixed_artifacts/boozer_unnest_reconstruct_20260820_tmp_originals/`.

GATE 1 (`tests/jax/objectives/test_flat675_objective.py`) is **not** this
check. GATE 1 is flat-vs-flat objective / QR `y` vs that archived native
*initial* certificate. Reconstruct is: freeze coils, take un-nest `s`, run
C++ `BoozerSurface` Newton, see if `s` moves.

---

## 0. Vocabulary (required)

| Name | What it is |
| --- | --- |
| **Nested Boozer** | Outer `x` = coils. Inner solve for `(s, ι, G)` so the surface is a Boozer surface of *these* coils, then score QS / ι / geometry. Banana production. |
| **LS** | `constraint_weight=1.0` on the **255×64** grid (`weight_inv_modB=True`). Stationarity is `∇_{s,ι,G} J_LS ≈ 0`, not `‖r‖ = 0`. Volume **penalized**. Banana default. `J_LS` below is the C++ Boozer LS scalar from `boozer_penalty_constraints_vectorized`, **not** the eight-term F3 outer `J`. |
| **Exact** | `constraint_weight=None` on the **21×21** collocation grid (`2 ntor+1` × `2 mpol+1`). Square `r = 0` (unweighted residual) plus `V = V_target` **enforced**. Illegal on 255×64 for stellsym (`get_stellsym_mask`). |
| **Un-nest / flat-675 / F3** | Outer `x` = 675 = 11 coil + 3 vessel + 661 surface Fourier. Inner `(ι, G)` by 48960×2 QR. Boozer residual is a **term in J** (weight 1000). Unconstrained L-BFGS-B. |
| **Purpose** | QS / ι / volume are properties of a Boozer surface of these coils. |
| **Journey** | How you get there (nested Newton iterates vs QR vs fused L-BFGS-B vs KKT). |
| **Reconstruct** | Freeze coils, start from un-nest `s` + QR `(ι, G)`, run C++ Newton of the chosen mode. Match ⇒ Newton is a no-op. |

Two modes ⇒ two judges. Do not score exact designs with the LS judge or LS designs with the exact judge.

---

## 1. Protocol

For each point:

1. Materialize native `BiotSavart` + `SurfaceXYZTensorFourier` from the frozen
   payload and the 675 candidate (fair-bar `Fullspace675NativeBoozerSystemMaterializer`).
2. Un-nest inner: reduced QR on the attested 48960×2 `(A, b)`.
3. **LS judge:** C++ `minimize_boozer_penalty_constraints_newton`
   (`constraint_weight=1.0`, `weight_inv_modB=True`, `stab=1e-4`, `tol=1e-13`,
   `maxiter=10`, `G` free). Coils must not move (`‖Δx_coils‖_∞ = 0`).
4. **Exact on 255×64:** expect the stellsym collocation exception.
5. **Exact on 21×21:** copy the same Fourier dofs onto
   `φ = linspace(0, 1/nfp, 21, endpoint=False)`,
   `θ = linspace(0, 1, 21, endpoint=False)`
   (drivers and `solve_residual_equation_exactly_newton` docstring; omitting
   `endpoint=False` is a different grid).
   Run A calls the library `solve_residual_equation_exactly_newton`
   (`maxiter=20`). Run B is a NumPy Newton loop on C++
   `boozer_surface_residual` (`maxiter=40`) with the same persist predicate
   (`_boozer_iterate_is_persistable`). Failed iterates **roll back**;
   `‖Δs‖ = 0` after failure is **not** a match.
6. **LS→exact handoff:** take the LS-polished Fourier dofs, rebuild 21×21, exact Newton.

Points:

| Label | Source |
| --- | --- |
| **archived start** | Bundle start candidate (GATE 1 / archived `y_certificate`) |
| **F3 B37 endpoint** | Fused GPU `pair2-l1` `endpoint_candidate` after `nit=37`, `nfev=48` |

Surface metadata at both points: `mpol=ntor=10`, `nfp=5`, stellsym, 661 surface
DOFs, LS grid 255×64, Volume target `0.1`, ι target `0.15`.

---

## 2. Experiment A — LS reconstruct (C++ LS Newton polish)

### 2.1 Archived start — **LS match**

| Quantity | Value |
| --- | --- |
| QR `(ι, G)` | `0.1500517839808274`, `2.010619295609829` |
| QR `‖Ay−b‖₂` | `3.4839688084386477e-3` (rel `3.000926282712338e-3`) |
| Check | `255×64×3 = 48960` residual rows; QR `A` shape `[48960, 2]` |
| Volume (255×64) | `0.09988741325948698` (`V − 0.1 = −1.1258674051302375e-4`) |
| LS `‖∇J‖_∞` before | `4.36751871647767e-15` |
| C++ LS Newton | **success, iter = 0** |
| `Δι`, `ΔG`, `‖Δs‖₂`, `‖Δs‖_∞` | **all 0** |
| Coils | frozen |
| Newton wall | `16.86650273296982` s (reconstruct `ls_newton.seconds`) |
| Jacobian probes (timed separately) | `1.15978` s before + `0.20429` s after |

QR `(ι, G)` match the archived native initial `y_certificate` exactly.
QR `‖Ay−b‖₂` is `3.4839688084386477e-3` vs certificate `3.4839688084386472e-3`
(1 ULP). Nested LS Newton is a no-op: the start already sits on the
LS-critical manifold.

**Match is `∇J_LS ≈ 0`, not `‖r‖ ≈ 0`.** The QR residual is still `3.5e-3` at a
machine-zero LS gradient. Overdetermined LS cannot zero the residual.

### 2.2 F3 B37 fused endpoint — **LS does not match**

| Quantity | Un-nest (F3) | After C++ LS Newton |
| --- | --- | --- |
| `(ι, G)` | `0.1516496147846734`, `2.0106192982546816` | `0.1408571095530796`, `2.0106193053897154` |
| QR `‖Ay−b‖₂` | `4.994314449402845e-3` (rel `4.3018418488001585e-3`) | (not re-QR’d) |
| Volume | `0.0998182004600127` | `0.09992911877773755` |
| LS `‖∇J‖₂` (C++ `norm` / success) | — | `3.9396376879063985e-14` (`≤ 1e-13`) |
| LS `‖∇J‖_∞` (report-only here) | `3.7103461544906323e-3` | `3.6163839464303325e-14` |
| C++ `J_LS` (not F3 outer `J`) | `1.2488312566898948e-5` | `8.860896032364874e-6` |
| Newton | — | **success** by C++ `‖∇J‖₂ ≤ 1e-13`; `iter=10`, `maxiter=10` |
| `Δι` | — | **`−0.01079250523159378`** |
| `‖Δs‖₂` / `‖Δs‖_∞` | — | **`8.806388241728436e-3` / `5.035304667539209e-3`** |
| Coils | frozen | frozen |
| Wall | — | **159.5 s** |

Un-nest QR vs the lane `endpoint_inner_state`
`[0.1516496147846736, 2.010619298254679]` is **8 ULP in ι** and **6 ULP in G**
(`≲1.5e-15` relative). That is not the 1-ULP residual gap of §2.1.

The walk is **not** ULP. F3’s sealed B37/BQ oracle-relative between fused and
native *lane* endpoints is `−2.25e-11`
(`docs/receipts/flat675_fused_campaign.md`). Reconstruct moves ι by `0.01079`
(~7.2% of the 0.15 target). `‖Δs‖_∞ = 5.035e-3` is the largest
**Fourier-coefficient** change in metres (Cartesian `SurfaceXYZTensorFourier`
dofs), not a measured pointwise surface displacement. Nested LS from this `s`
is a **different design** (ι `0.15165 → 0.14086`). The 661 LS-projected
Fourier dofs are in
`docs/receipts/evidence/boozer_unnest_newton_reconstruct_20260820/boozer_ls_exact_purpose.json`
at `points[name=f3_b37_pair2_l1_endpoint].ls.surface_dofs`. The eight-term
outer `J` was not re-scored after the walk.

Purpose of nested LS: held at start; **not** held at the F3 endpoint. Residual
weight 1000 was not enough against ι-penalty / QS / geometry terms. Un-nest
used the extra `s` freedom to keep ι near 0.15; nested LS would have projected
back onto the Boozer LS manifold every eval.

---

## 3. Experiment B — exact on the 675 residual grid

Both points: **skipped**.

```
Exception: Stellarator symmetric BoozerExact surfaces require a specific set of
quadrature points … SurfaceXYZTensorFourier.get_stellsym_mask()
```

Grid was 255×64, stellsym true. Exact **cannot** be the F3 operator.

---

## 4. Experiment C — exact Newton on the collocation grid (21×21)

Fourier dofs copied onto the legal collocation quadrature. Check:
`2·ntor+1 = 2·mpol+1 = 21`. First run used the library
`solve_residual_equation_exactly_newton` (`maxiter=20`). Second run logged a
**NumPy** Newton loop on C++ `boozer_surface_residual` (`maxiter=40`) and
applied `_boozer_iterate_is_persistable`. Exact residual is **unweighted**
(`weight_inv_modB=False`); LS reconstruct used `True`.

### 4.1 Start (already LS-perfect)

| | Library (`maxiter=20`) | Logged loop (`maxiter=40`) |
| --- | --- | --- |
| Initial unmasked `‖r‖_∞` | `6.861477038507605e-3` | same |
| Initial masked `‖b‖₂` (residual + label) | — | `3.3216550932751505e-2` |
| Volume on 21×21 | `0.09988714921610342` | same |
| Success / persist | false / rolled back | false / rolled back |
| `‖Δs‖` after rollback | 0 | 0 |
| Wall | 1.00 s | 1.80 s |

Logged Newton **diverges on step 1** (start, LS→exact identical because LS
iter=0):

| iter | ι | masked `‖b‖₂` | volume |
| --- | ---: | ---: | ---: |
| 0 | 0.15005 | 0.0332 | 0.09989 |
| 1 | 0.1920 | **5.24** | 0.125 |
| 2 | 0.1933 | 1.57 | 0.103 |
| 39 | **−20.30** | **~2×10⁷** | ~10⁷ |

Minimum masked residual over 40 steps equals the **initial** residual. Then
rollback. `‖Δs‖ = 0` here is **not** agreement.

### 4.2 F3 endpoint

Un-nest `s` on 21×21: initial masked `‖b‖₂ = 0.04671936870042946`, unmasked
`‖r‖_∞ = 0.008648739458689568` (library reconstruct and logged loop agree on
`‖r‖_∞`). Logged loop: success false, persist false, min residual = initial,
rollback.

LS-polished F3 `s` (the walked ι=0.141 surface) as exact seed — next section.

---

## 5. Experiment D — LS→exact handoff (production recipe)

Copy **LS-polished** Fourier dofs onto 21×21, exact Newton from LS `(ι, G)`.

| Seed | Initial masked `‖b‖₂` | iter 1 | Outcome |
| --- | ---: | --- | --- |
| Start (LS = un-nest) | 0.0332 | 5.24, ι 0.192 | diverge, rollback |
| F3 after LS project | **0.03927** | **2.26**, ι **0.217**, V 0.133 | diverge, rollback |

F3 handoff is only slightly better than exact-from-un-nest (`0.03927` vs
`0.04672`) and still explodes. Exact is **not** “LS, but tighter.” Different
grid, unweighted square residual, volume enforced. This 255×64 LS surface is
**not in the exact basin**.

---

## 6. What “both supported” means after the measurements

| Mode | Nested banana | Flat-675 / F3 | Purpose without nested-in-outer |
| --- | --- | --- | --- |
| **LS** | yes (`constraint_weight=1.0`, 255×64) | yes (QR + residual in `J`) | **yes if on-manifold**: start yes; F3 only after LS *project* (changes the design) |
| **Exact** | yes (`None`, **rebuilds 21×21**) | **no** | **not from these states** |

Supporting both is two contracts (two grids, two judges, two inits), not a bool
on F3.

A **fused exact lane** (collocation `s` in `x`, square `r=0` as equalities, KKT
in XLA) would be a **new formulation**: new residual, constrained solver (not
unconstrained L-BFGS-B), new init (must start already exact), new C++ bar.
It was **not implemented**. Nested C++ exact on 21×21 is already ~1 s per
reconstruct; F3 fused still pays a ~34–35 s per-process floor
(`docs/receipts/flat675_fused_campaign.md`). There is no measured reason to
expect an F3-like GPU× versus C++ nested exact.

---

## 7. Public GitHub — GPU exact BoozerSurface?

Survey (2026-08-20), code search + repo READMEs. Two different “Boozers”:

| Project | GPU? | Exact coil-field `r=0` Newton? |
| --- | --- | --- |
| [hiddenSymmetries/simsopt](https://github.com/hiddenSymmetries/simsopt) | C++ CPU | **yes** (`solve_residual_equation_exactly_newton`) |
| [andrewgiuliani/PySurfaceOpt](https://github.com/andrewgiuliani/PySurfaceOpt) | CPU via simsopt | **yes** (nested QS on those surfaces) |
| [uwplasma/booz_xform_jax](https://github.com/uwplasma/booz_xform_jax) | JAX, GPU-capable | **no** — VMEC *transform* |
| [uwplasma/vmec_jax](https://github.com/uwplasma/vmec_jax) | JAX GPU-capable | **no** — MHD; QI uses xform |
| [PlasmaControl/DESC](https://github.com/PlasmaControl/DESC) | JAX GPU-capable | **no** — Boozer transform / QS on a DESC eq |
| [uwplasma/ESSOS](https://github.com/uwplasma/ESSOS) | JAX coils | **no** — SIMSOPT VMEC Boozer *xform* |
| SIMSOPT PRs [#604](https://github.com/hiddenSymmetries/simsopt/pull/604), [#623](https://github.com/hiddenSymmetries/simsopt/pull/623) | JAX wrappers | **no** — `booz_xform_jax`, not `BoozerSurface` exact |

Public `BoozerSurfaceJAX` / `solve_residual_equation_exactly` on GPU: **no hits**.
GPU Boozer stacks transform equilibria. Exact-surface stacks are CPU SIMSOPT.

Private `BoozerSurfaceJAX` exact inner on GPU is documented in this tree at
`docs/receipts/custom-quasi-newton/boozer-exact-inner-native-scale-cpu-gpu-20260802/summary.md`
(source commit `19194b957`, **diagnostic-pass-not-promotion**): CPU residual
`1.98e-29`, RTX 5090 `2.52e-29`; both runs accepted **zero outer steps**.
Nested inner-solve evidence, not a fused exact speed win. That receipt is not
this reconstruct and is not F3 production `580217e0c`.

---

## 8. Recommendation (from these experiments)

1. **Cite F3 as LS-flat GPU vs native.** Do not call it nested banana or exact.
2. **Do not build fused exact as the next GPU campaign** unless the science
   requires `r=0` *and* many evals; the inner is already cheap and the fused
   process floor is large.
3. **If nested purpose is required on a flat design:** freeze coils, C++ LS
   Newton, gate `|Δι|` / `‖Δs‖_∞` / C++ `‖∇J‖₂`. Fail ⇒ project (publish the
   projected surface — for this F3 point, the 661 dofs in the purpose JSON)
   or tighten the LS constraint and re-optimize. Do **not** exact-polish F3
   states.
4. **Keep exact nested** on 21×21 (C++ or JAX `run_code`). Two software
   contracts if both modes must exist.

Default path: GPU stays on **LS-flat**; reconstruct is a **physics gate** on
endpoints; nest only when that gate fails and the nested optimum is actually
needed.

---

## 9. This note does not claim

- Any change to F3 pair ratios (1.67× / 7.70× / 7.36× process wall).
- That un-nest equals nested after optimization.
- That QR residual ~0 is the LS match criterion.
- That exact-Newton `‖Δs‖=0` after rollback is a match.
- A GPU exact fused win, or a public GPU exact `BoozerSurface`.
- Re-evaluation of the eight-term outer `J` at the LS-projected F3 point
  (not run).
- Full banana `run_code` LS path (BFGS 1500 + Newton 50). Newton polish is
  the “already near a root?” probe; at F3, Newton alone reached `∇J ~ 1e-14`.
