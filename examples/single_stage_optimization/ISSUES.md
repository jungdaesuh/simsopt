# Single-Stage Optimization Workflow: Issue Tracker

Comprehensive code review covering `banana_coil_solver.py`, `single_stage_banana_example.py`,
`poincare_surfaces.py`, and associated shell scripts.
Review date: 2026-03-14.

Candidate local status update: on 2026-03-16, candidate-only fixes were applied and validated
in the local working tree. **Batch 1:** [Issue 1](#1), [Issue 2](#2), [Issue 3](#3),
[Issue 4](#4), [Issue 7](#7), and [Issue 19](#19) in
`single_stage_banana_example.py` and `poincare-plot.sh`. Issue 2 fix validated via empirical
testing (20D Rosenbrock against scipy L-BFGS-B) and a regression test in
`tests/geo/test_single_stage_example.py`. **Batch 2:** [Issue 5](#5), [Issue 6](#6) (segment
distance algorithm rewrite with Sunday/Lumelsky), [Issue 8](#8), [Issue 9](#9) (cross-section
angle normalization), and [Issue 31](#31) (ftol/gtol None crash) in `banana_coil_solver.py` and
`single_stage_banana_example.py`. Segment distance fix validated against 8 named test cases
plus 1000-pair random brute-force. All Batch 2 fixes covered by committed regression tests in
`tests/geo/test_single_stage_example.py` (segment distance tests extract the real deployed
function via AST; cross-section and ftol/gtol tests use source-level and module-level
assertions). `ftol_by_mpol`/`gtol_by_mpol` moved to module scope for testability.
**Batch 3:** [Issue 25](#25), [Issue 26](#26) (dead `hbt_poly` variables),
[Issue 27](#27), [Issue 28](#28), [Issue 29](#29) (unused imports) in
`banana_coil_solver.py`, `single_stage_banana_example.py`, and `poincare_surfaces.py`.
Pure deletion of dead code and unused imports; validated via `py_compile` and reviewer agent.
**Batch 4:** Near-parallel interior minimum fix for `segment_segment_distance` in
`banana_coil_solver.py`. The near-parallel branch (denom < PAR_EPS * a * c) previously only
checked 4 endpoint projections, missing interior minima where both parameters are in (0,1).
Added an interior check using the general-case formula — well-conditioned when the true minimum
is interior because numerators scale with the small denominator. Validated against 50K adversarial
near-parallel pairs (0 failures). Two new regression tests added to
`tests/geo/test_single_stage_example.py`: deterministic adversarial case (9x overestimate
before fix) and 1000-pair near-parallel brute-force.
**Batch 5:** [Issue 10](#10) (VTK label fix), [Issue 13](#13) (initSurface parameterized),
[Issue 14](#14) (initializeCoils parameterized), [Issue 15](#15) (normPlot global reshape removed),
[Issue 16](#16) (magneticFieldPlots global reshape removed), [Issue 17](#17) (endpoint quadrature),
[Issue 21](#21) (set_points before reshape), [Issue 22](#22) (poincare OUT_DIR),
[Issue 30](#30) (Stage 2 tol=1e-15 replaced with explicit --ftol/--gtol CLI flags), [Issue 32](#32) (leaked figure), [Issue 33](#33) (tick labels),
[Issue 34](#34) (blank subplots), [Issue 36](#36) (crossSectionPlot/fun global deps parameterized)
in `banana_coil_solver.py`, `single_stage_banana_example.py`, and `poincare_surfaces.py`.
All reshape operations now derive dimensions from the surface instead of global `nphi`/`ntheta`.
Validated via `py_compile` and reviewer agent (2 rounds).
**Batch 6:** [Issue 24](#24) (`__name__` guards added to `banana_coil_solver.py` and `poincare_surfaces.py`),
[Issue 35](#35) (shell scripts use configurable `SIMSOPT_ROOT` and `CONDA_ENV` variables).
Stage 2 output directory float formatting aligned with single-stage lookup pattern (`:g` format).
Bonus fixes: `success1` pre-initialized in single-stage `fun()` to prevent `UnboundLocalError`;
`crossSectionPlot` now plots banana coil R-Z projection (implementing the original author's
unfinished intent — parameter was passed but never used upstream).
Unless otherwise noted, the issue descriptions below still describe the pre-fix audit state.

---

## Severity Legend

| Tag | Meaning |
|-----|---------|
| **C** | **Critical** -- wrong results, crash, or data corruption |
| **M** | **Moderate** -- misleading output or fragile design that will break under reasonable use |
| **L** | **Low** -- cleanup, dead code, or cosmetic issue |

## Validation Status Legend

| Status | Meaning |
|--------|---------|
| **Active** | Broken or misleading in the current code path without extra assumptions |
| **Latent / design** | Real issue, but conditional, architectural, or mostly exposed by reuse / changed inputs |
| **Conditional** | Only manifests for certain configurations not currently exercised by the script defaults |
| **Not a bug** | Reviewed and intentionally correct, or cache-safe enough to not count as a defect |

---

## Table of Contents

| # | Sev | File | Lines | Short Title |
|---|-----|------|-------|-------------|
| [1](#1)  | C | single_stage | ~~346, 364, 379~~ | ~~Missing imports crash `--boozer-stage final`~~ :white_check_mark: |
| [2](#2)  | C | single_stage | ~~567-568~~ | ~~Negated gradient can corrupt L-BFGS-B Hessian~~ :white_check_mark: |
| [3](#3)  | C | single_stage | ~~621-622~~ | ~~Swapped Cartesian axes in cylindrical R/Z~~ :white_check_mark: |
| [4](#4)  | C | single_stage | ~~629, 781, 922~~ | ~~`intersecting` never updates (Python scope)~~ :white_check_mark: |
| [5](#5)  | C | banana_coil | ~~240-242~~ | ~~Segment distance missing re-projection after clamping~~ :white_check_mark: |
| [6](#6)  | C | banana_coil | ~~233-235~~ | ~~Parallel-segment case incomplete~~ :white_check_mark: |
| [7](#7)  | C | poincare sh | ~~33~~ | ~~Shell script references wrong Python filename~~ :white_check_mark: |
| [8](#8)  | M | banana_coil | ~~360-362~~ | ~~Cross-section angle double-scaled~~ :white_check_mark: |
| [9](#9)  | M | single_stage | ~~493-495~~ | ~~Cross-section angle double-scaled (duplicate)~~ :white_check_mark: |
| [10](#10) | M | banana_coil | ~~524~~ | ~~VTK export labeled "VV" is actually banana coil surface~~ :white_check_mark: |
| [11](#11) | M | banana_coil | 500 | Output dir omits run-defining parameters (deferred — format aligned with `:g`) |
| [12](#12) | M | single_stage | 755 | Output dir only encodes mpol/ntor (deferred — results.json captures all) |
| [13](#13) | M | banana_coil | ~~170-178~~ | ~~`initSurface` depends on globals `file_loc`, `nphi`, `ntheta`~~ :white_check_mark: |
| [14](#14) | M | banana_coil | ~~180-201~~ | ~~`initializeCoils(surf)` ignores its argument for coil placement~~ :white_check_mark: |
| [15](#15) | M | single_stage | ~~464~~ | ~~`normPlot` reshapes using global `nphi`/`ntheta`~~ :white_check_mark: |
| [16](#16) | M | banana_coil | ~~317~~ | ~~`magneticFieldPlots` reshapes using global `nphi`/`ntheta`~~ :white_check_mark: |
| [17](#17) | M | banana_coil | ~~182~~ | ~~Quadrature includes duplicate endpoint~~ :white_check_mark: |
| [18](#18) | -- | single_stage | 300-301 | ~~`BoozerResidualExact` hardcodes `constraint_weight=0`~~ (by design) |
| [19](#19) | C | single_stage | ~~417-418~~ | ~~Exact Boozer quadpoints use `mpol` for phi -- throws for `mpol != ntor`~~ :white_check_mark: |
| [20](#20) | -- | single_stage | 594-595 | ~~`callback` re-calls JF.J()/JF.dJ()~~ (cache-safe) |
| [21](#21) | M | single_stage | ~~769-770~~ | ~~`bs.B()` reshape assumes nphi*ntheta before `set_points`~~ :white_check_mark: |
| [22](#22) | M | poincare | ~~94~~ | ~~Hardcoded relative `OUT_DIR`~~ :white_check_mark: |
| [23](#23) | M | cross-file | -- | Duplicated `crossSectionPlot`/`normPlot` (deferred — now fully parameterized) |
| [24](#24) | M | cross-file | -- | ~~No `if __name__ == "__main__"` guard in any script~~ :white_check_mark: |
| [25](#25) | L | banana_coil | ~~354~~ | ~~Dead `hbt_poly` variable~~ :white_check_mark: |
| [26](#26) | L | single_stage | ~~487~~ | ~~Dead `hbt_poly` variable (duplicate)~~ :white_check_mark: |
| [27](#27) | L | banana_coil | ~~13-22~~ | ~~Unused imports~~ :white_check_mark: |
| [28](#28) | L | single_stage | ~~6, 16, 17~~ | ~~Unused imports (`Polygon`, `save`, `ScaledCurrent`)~~ :white_check_mark: |
| [29](#29) | L | poincare | ~~4-5, 9~~ | ~~Unused imports (`SurfaceClassifier`, `LevelsetStoppingCriterion`, `Line2D`)~~ :white_check_mark: |
| [30](#30) | L | banana_coil | ~~509~~ | ~~`tol=1e-15` is exceptionally strict~~ :white_check_mark: |
| [31](#31) | L | single_stage | ~~852-853~~ | ~~`ftol`/`gtol` returns `None` for out-of-range `mpol` (crashes)~~ :white_check_mark: |
| [32](#32) | L | poincare | ~~30-31~~ | ~~Leaked matplotlib figure~~ :white_check_mark: |
| [33](#33) | L | poincare | ~~47~~ | ~~Y-axis tick labels only hidden for column 1~~ :white_check_mark: |
| [34](#34) | L | poincare | ~~29~~ | ~~Blank subplots for non-square phi count~~ :white_check_mark: |
| [35](#35) | L | shell | ~~various~~ | ~~Shell scripts hardcoded to `~/simsopt/` and `conda activate simsopt`~~ :white_check_mark: |
| [36](#36) | L | banana_coil | ~~377-388~~ | ~~`fun()` closes over many globals~~ :white_check_mark: (factory pattern + banana coil R-Z plot) |
| [37](#37) | -- | banana_coil | 81 | ~~Default `s=0.24` is an interior VMEC surface~~ (intentional) |

---

## Tracking Checklist

Checklist meaning:

- `[x]` fixed locally in the candidate code
- `[ ]` still open

- [x] [Issue 1](#1) Missing imports crash `--boozer-stage final`
- [x] [Issue 2](#2) Negated gradient can corrupt L-BFGS-B Hessian
- [x] [Issue 3](#3) Swapped Cartesian axes in cylindrical R/Z
- [x] [Issue 4](#4) `intersecting` never updates
- [x] [Issue 5](#5) Segment distance missing re-projection after clamping
- [x] [Issue 6](#6) Parallel-segment case incomplete
- [x] [Issue 7](#7) Shell script references wrong Python filename
- [x] [Issue 8](#8) Cross-section angle double-scaled
- [x] [Issue 9](#9) Cross-section angle double-scaled (duplicate)
- [x] [Issue 10](#10) VTK export labeled "VV" is actually banana coil surface
- [ ] [Issue 11](#11) Output dir omits run-defining parameters (deferred — format aligned)
- [ ] [Issue 12](#12) Output dir only encodes `mpol`/`ntor` (deferred — results.json captures all)
- [x] [Issue 13](#13) `initSurface` depends on globals
- [x] [Issue 14](#14) `initializeCoils(surf)` ignores its argument
- [x] [Issue 15](#15) `normPlot` reshapes using global `nphi`/`ntheta`
- [x] [Issue 16](#16) `magneticFieldPlots` reshapes using global `nphi`/`ntheta`
- [x] [Issue 17](#17) Quadrature includes duplicate endpoint
- [ ] [Issue 18](#18) `BoozerResidualExact` hardcodes `constraint_weight=0` (reviewed, by design)
- [x] [Issue 19](#19) Exact Boozer quadpoints use `mpol` for phi
- [ ] [Issue 20](#20) `callback` re-calls `JF.J()/JF.dJ()` (reviewed, cache-safe)
- [x] [Issue 21](#21) `bs.B()` reshape assumes `nphi*ntheta` before `set_points`
- [x] [Issue 22](#22) Hardcoded relative `OUT_DIR`
- [ ] [Issue 23](#23) Duplicated `crossSectionPlot`/`normPlot` (deferred — now parameterized)
- [x] [Issue 24](#24) No `if __name__ == "__main__"` guard in any script
- [x] [Issue 25](#25) Dead `hbt_poly` variable
- [x] [Issue 26](#26) Dead `hbt_poly` variable (duplicate)
- [x] [Issue 27](#27) Unused imports
- [x] [Issue 28](#28) Unused imports (`Polygon`, `save`, `ScaledCurrent`)
- [x] [Issue 29](#29) Unused imports (`SurfaceClassifier`, `LevelsetStoppingCriterion`, `Line2D`)
- [x] [Issue 30](#30) `tol=1e-15` is exceptionally strict
- [x] [Issue 31](#31) `ftol`/`gtol` returns `None` for out-of-range `mpol`
- [x] [Issue 32](#32) Leaked matplotlib figure
- [x] [Issue 33](#33) Y-axis tick labels only hidden for column 1
- [x] [Issue 34](#34) Blank subplots for non-square phi count
- [x] [Issue 35](#35) Shell scripts hardcoded to `~/simsopt/` and `conda activate simsopt`
- [x] [Issue 36](#36) `fun()` closes over many globals (factory pattern; banana coil R-Z projection implemented)
- [ ] [Issue 37](#37) Default `s=0.24` is an interior VMEC surface (reviewed, intentional)

---

## Critical Issues

<a id="1"></a>
### 1. [C] Missing imports crash `--boozer-stage final`

**File:** `single_stage_banana_example.py`
**Lines:** 346, 364, 379

`BoozerResidualExact.compute()` and `dJ_by_dB()` call three functions that are never imported:

```python
r, J = boozer_surface_residual(surface, iota, G, ...)          # line 346
adj = forward_backward(P, L, U, dJ_ds)                        # line 364
r, r_dB = boozer_surface_residual_dB(surface, ..., ...)        # line 379
```

These exist in the SIMSOPT codebase:

- `boozer_surface_residual` -- `simsopt.geo.surfaceobjectives`
- `boozer_surface_residual_dB` -- `simsopt.geo.surfaceobjectives`
- `forward_backward` -- `simsopt.objectives.utilities`

Running `--boozer-stage final` will hit `NameError` at the first optimizer iteration.

**Fix:** Add imports:
```python
from simsopt.geo.surfaceobjectives import boozer_surface_residual, boozer_surface_residual_dB
from simsopt.objectives.utilities import forward_backward
```

**Candidate update (2026-03-16):** Fixed locally in the candidate working tree by adding the
missing imports. Validated with `python -m py_compile`, a targeted candidate
`--boozer-stage final --init-only` run after [Issue 19](#19) was also fixed, and a dedicated
regression test in
`/Users/suhjungdae/code/columbia/simsopt/tests/geo/test_single_stage_example.py`. This note
applies to the local candidate only, not the historical baseline.

---

<a id="2"></a>
### 2. [C] Negated gradient can corrupt L-BFGS-B Hessian approximation

**File:** `single_stage_banana_example.py`
**Lines:** 567-568

```python
else:  # Boozer surface failed
    J = run_dict['J']       # same J as last accepted
    dJ = -run_dict['dJ']    # NEGATED gradient
```

When the Boozer surface solve fails, the optimizer receives identical `J` but a flipped gradient
direction. This feeds an inconsistent `(J, dJ)` pair to L-BFGS-B. Empirical analysis (20D
Rosenbrock, verified against scipy 1.17.1 `_lbfgsb_py.py` reverse-communication loop) shows
the actual failure mode depends on whether failures are transient or persistent:

- **Transient failures** (Boozer fails at full step but succeeds at smaller step): the
  Moré-Thuente line search rejects the fake evaluation (sufficient decrease `f_new ≤ f_old +
  c₁·α·gd` fails because `f_new = f_old`), backtracks, and the next evaluation at a smaller
  step succeeds. Minimal impact (~5-10% more function evaluations).

- **Persistent failures** (Boozer fails for ALL step sizes in the line search): the line search
  exhausts `maxls` attempts, L-BFGS-B restores `(x, f, g)` to the previous iterate, **flushes
  all BFGS correction pairs** (memory reset to identity), and emits
  `ABNORMAL_TERMINATION_IN_LNSRCH`. The Hessian is destroyed, not corrupted. If Boozer
  continues to fail, the optimizer terminates abnormally.

The BFGS curvature check (`y^T s > 0`) is not reached with the fake values in the normal path
because the line search rejects the step before the BFGS update. However, the `(J, dJ)` pair is
mathematically inconsistent regardless.

**Fix:** Return elevated objective with the last accepted gradient (same sign, not negated):
```python
J = run_dict['J'] + max(abs(run_dict['J']), 1.0)
dJ = run_dict['dJ'].copy()
```

This fix provides two guarantees:
1. **Elevated J** triggers line search backtracking (sufficient decrease fails convincingly).
2. **Stale gradient (same sign)** ensures `y_k = g_new - g_old = 0` if the step is accepted
   via the Moré-Thuente `WARN` path, so the L-BFGS-B curvature check
   (`dr ≤ epsmch·rr` with `dr=0, rr=0`) safely skips the BFGS Hessian update.

Note: returning `(J_old + LARGE_PENALTY, zero_gradient)` is **not recommended** — empirical
testing shows the zero gradient produces `y_k = -dJ_old` which can pass the curvature check
(`y^T s > 0`) for descent directions, and an oversized penalty can dominate the landscape and
cause the optimizer to converge to a wrong point.

**Candidate update (2026-03-16):** Fixed locally in the candidate working tree.

---

<a id="3"></a>
### 3. [C] Swapped Cartesian axes in cylindrical R/Z callback diagnostic

**File:** `single_stage_banana_example.py`
**Lines:** 621-622

```python
max_r = np.max(np.sqrt(banana_curve.gamma()[:,1]**2 + banana_curve.gamma()[:,2]**2))  # y^2 + z^2
max_z = np.max(np.abs(banana_curve.gamma()[:,0]))                                      # x
```

SIMSOPT `curve.gamma()` returns `[x, y, z]` Cartesian (confirmed from `Curve.plot()` in
`simsopt/src/simsopt/geo/curve.py`). Cylindrical coordinates are `R = sqrt(x^2 + y^2)`,
`Z = z`. The code computes:

- `max_r` from `sqrt(y^2 + z^2)` -- **wrong plane**
- `max_z` from `abs(x)` -- **wrong axis**

The `Max Curve R` and `Max Curve Z` diagnostics logged to `log.txt` are **meaningless** for the
entire optimization run.

**Fix:**
```python
max_r = np.max(np.sqrt(banana_curve.gamma()[:,0]**2 + banana_curve.gamma()[:,1]**2))
max_z = np.max(np.abs(banana_curve.gamma()[:,2]))
```

---

<a id="4"></a>
### 4. [C] `intersecting` flag never updates in outer scope

**File:** `single_stage_banana_example.py`
**Lines:** 629, 781, 922

```python
intersecting = False                                          # line 781 -- module scope

def callback(x):
    ...
    intersecting = boozer_surface.surface.is_self_intersecting()  # line 629 -- LOCAL variable
```

Python assignment inside `callback` creates a local variable that shadows the module-level one.
The module-level `intersecting` stays `False` forever. `results.json` always reports
`"SELF_INTERSECTING": false` regardless of actual state.

**Fix:** Store in `run_dict` instead:
```python
# In callback:
run_dict['intersecting'] = boozer_surface.surface.is_self_intersecting()

# In results dict:
"SELF_INTERSECTING": run_dict.get('intersecting', False),
```

---

<a id="5"></a>
### 5. [C] Segment-segment distance missing re-projection after clamping

**File:** `banana_coil_solver.py`
**Lines:** 240-242

```python
# scalar-safe clipping
s = 0.0 if s < 0.0 else (1.0 if s > 1.0 else s)
t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
```

The segment-segment distance function clamps both `s` and `t` independently without
re-projecting. The correct algorithm (Sunday/Lumelsky) requires that after clamping one
parameter to a boundary, the other must be recomputed by projecting the clamped point onto the
other segment.

**Impact:** The function **overestimates** distances. For a minimum-distance safety constraint,
this is the dangerous direction -- the self-intersection checker at line 516 may **fail to detect
self-intersecting coils**. Demonstrated errors:

- ~5.4% for simple skew segments (P1=(0,0,0) P2=(2,1,0) vs Q1=(-1,3,0) Q2=(1,2,0):
  buggy=1.414, correct=1.342)
- ~8x for parallel overlapping segments (P along x [0,10], Q parallel at y=1 from x=[8,20]:
  buggy=8.06, correct=1.0)

This bug does NOT affect the coil-coil distance in the objective function (`Jccdist` uses
SIMSOPT's built-in `CurveCurveDistance`), only the post-optimization `is_self_intersecting`
validation.

**Fix:** Implement the full Sunday algorithm with re-projection after each clamp boundary, or
use a reference implementation.

**Candidate update (2026-03-16):** Fixed locally by rewriting `segment_segment_distance` with
the full Sunday/Lumelsky algorithm. Changes: (1) degenerate segment handling (zero-length),
(2) relative parallelism threshold (`PAR_EPS * a * c` instead of absolute `1e-14`),
(3) near-parallel case checks all four endpoint-to-segment projections,
(4) general case re-projects the other parameter after each clamp. Committed regression tests
in `tests/geo/test_single_stage_example.py` extract the real deployed function via AST and
validate against 8 named cases + 1000-pair random brute-force (interior + edge exhaustive
search).

---

<a id="6"></a>
### 6. [C] Parallel-segment case incomplete

**File:** `banana_coil_solver.py`
**Lines:** 233-235

```python
if denom < SMALL_NUM:
    s = 0.0
    t = e / c if c > SMALL_NUM else 0.0
```

When segments are nearly parallel (`denom < SMALL_NUM`), the code only checks the projection of
`P1` onto segment Q. For overlapping parallel segments, the minimum distance could occur at the
projection of `P2`, `Q1`, or `Q2` instead. Combined with the missing re-projection from
[Issue 5](#5), this produces catastrophically wrong results for overlapping parallel segments.

Additionally, the parallelism check uses an **absolute** epsilon (`1e-14`) instead of a relative
one (`SMALL_NUM * a * c`), making it scale-dependent. Since `denom = a*c*sin^2(theta)`:
**short** non-parallel segments can be falsely flagged as parallel (small `a*c` makes `denom`
tiny even at significant angles), while **long** near-parallel segments can slip through as
non-parallel (large `a*c` keeps `denom` above the threshold even at tiny angles).

**Candidate update (2026-03-16):** Fixed together with [Issue 5](#5) in the same rewrite.

---

<a id="7"></a>
### 7. [C] Poincare shell script references wrong filename

**File:** `POINCARE_PLOTTING/poincare-plot.sh`
**Line:** 33

```bash
python3 poincare-surfaces.py   # WRONG: hyphens
```

The actual Python file is `poincare_surfaces.py` (underscores). `sbatch poincare-plot.sh` will
crash with `FileNotFoundError` on a clean checkout.

**Fix:**
```bash
python3 poincare_surfaces.py
```

---

## Moderate Issues

<a id="8"></a>
### 8. [M] Cross-section angle double-scaled (Stage 2)

**File:** `banana_coil_solver.py`
**Lines:** 360-362

```python
phi_array = np.linspace(0, 2*np.pi / surf_coils.nfp * 4/5, 5)  # already in radians
for phi_slice in phi_array:
    cs = surf.cross_section(phi_slice * 2 * np.pi)               # radians * 2pi
```

`SurfaceRZFourier.cross_section(phi)` expects `phi` normalized by 2pi (i.e., range [0,1]).
The code passes values up to ~6.3 instead of ~0.16. The cross sections are plotted at completely
wrong toroidal angles, and the labels (`phi_slice/np.pi`) do not match.

Note: The Poincare script (`poincare_surfaces.py:67`) handles this correctly:
`phi_new = phis[i] * 1 / (2 * np.pi)`.

**Fix:**
```python
cs = surf.cross_section(phi_slice / (2 * np.pi))
```

**Candidate update (2026-03-16):** Fixed locally. Verified against SIMSOPT `Surface.cross_section`
docstring (phi normalized to [0,1]) and the correct usage in `poincare_surfaces.py:67`.

---

<a id="9"></a>
### 9. [M] Cross-section angle double-scaled (Single Stage)

**File:** `single_stage_banana_example.py`
**Lines:** 493-495

Identical bug to [Issue 8](#8), in the duplicated `crossSectionPlot` function.

```python
phi_array = np.linspace(0, 2*np.pi / surf_coils.nfp * 4/5, 5)
for phi_slice in phi_array:
    cs = surf.cross_section(phi_slice * 2 * np.pi)
```

**Fix:** Same as Issue 8.

**Candidate update (2026-03-16):** Fixed locally, same change as Issue 8.

---

<a id="10"></a>
### 10. [M] VTK export labeled "VV" is actually banana coil surface

**File:** `banana_coil_solver.py`
**Line:** 524

```python
new_surf_coils.to_vtk(OUT_DIR_ITER + "VV")
```

`new_surf_coils` is assigned from `surf_coils` (line 462), which is the banana coil winding
surface (R0=0.976, a=`banana_surf_radius`, default 0.22). The actual vacuum vessel is
defined as `VV` (R0=0.976, a=0.222) at lines 423-426 but is never exported. The output file
named `VV` contains the wrong geometry.

For default `banana_surf_radius=0.22`, the difference is only 0.002 m, so the visual impact in
ParaView is negligible. For non-default values, the mislabeling is genuinely misleading.

**Fix:**
```python
VV.to_vtk(OUT_DIR_ITER + "VV")
```

---

<a id="11"></a>
### 11. [M] Stage 2 output directory omits run-defining parameters

**File:** `banana_coil_solver.py`
**Line:** 500

```python
OUT_DIR_ITER = f"{OUT_DIR}R0={R0}-s={s}-LW={LENGTH_WEIGHT}-CCW={CC_WEIGHT}-CW={CURVATURE_WEIGHT}-SR={banana_surf_radius:0.3f}-Order={order}/"
```

Parameters encoded: `R0`, `s`, `LENGTH_WEIGHT`, `CC_WEIGHT`, `CURVATURE_WEIGHT`,
`banana_surf_radius`, `order`.

Parameters omitted: `CC_THRESHOLD`, `CURVATURE_THRESHOLD`, `LENGTH_TARGET`, `theta_center`,
`phi_center`, `theta_width`, `phi_width`, `nphi`, `ntheta`, `init_only`.

Two runs with the same weights but different thresholds or initialization angles silently
overwrite each other's `results.json`, `biot_savart_opt.json`, and diagnostic plots.

---

<a id="12"></a>
### 12. [M] Single-stage output directory only encodes mpol/ntor

**File:** `single_stage_banana_example.py`
**Line:** 755

```python
OUT_DIR_ITER = OUT_DIR + f"/mpol={mpol}-ntor={ntor}"
```

Only `mpol` and `ntor` are encoded. All of `iota_target`, `vol_target`, `boozer_stage`,
`cc_dist`, `cc_weight`, `curvature_weight`, `curvature_threshold`, `constraint_weight`,
Stage 2 seed identity, and `init_only` are omitted.

Partially mitigated by `--output-root` (users can set unique root directories per experiment),
but the default behavior allows distinct physics configurations to collide silently.

---

<a id="13"></a>
### 13. [M] `initSurface` depends on globals `file_loc`, `nphi`, `ntheta`

**File:** `banana_coil_solver.py`
**Lines:** 170-178

**Validation status:** Latent / design. Works in the current script order, but the function is not self-contained and is brittle under refactor or reuse.

```python
def initSurface(R0, s):
    surf = SurfaceRZFourier.from_wout(file_loc, range="full torus", nphi=nphi, ntheta=ntheta, s=s)
```

The function accepts `R0` and `s` as arguments but reads `file_loc`, `nphi`, and `ntheta` from
module-level globals defined at lines 397, 409-410. The function is not self-contained and will
break if refactored or called before globals are initialized.

---

<a id="14"></a>
### 14. [M] `initializeCoils(surf)` ignores its argument for coil placement

**File:** `banana_coil_solver.py`
**Lines:** 180-201

**Validation status:** Latent / design. The current script happens to call it consistently, but the function signature is misleading and tightly coupled to globals.

```python
def initializeCoils(surf):
    banana_curve = CurveCWSFourierCPP(..., surf=surf_coils)  # uses global surf_coils
    ...
    banana_coils = coils_via_symmetries(..., surf_coils.nfp, surf_coils.stellsym)  # global
    coils = tf_coils + banana_coils  # global tf_coils
```

The `surf` parameter is only used for field evaluation (`bs.set_points(surf.gamma()...)`), while
coil placement uses globals `surf_coils`, `tf_coils`, `OUT_DIR`, `nphi`, `ntheta`, `order`,
`phi_center`, `theta_center`, `phi_width`, `theta_width`, `num_quadpoints`. The interface is
misleading: the function signature suggests coils are placed on `surf`, but they are placed on
`surf_coils`.

---

<a id="15"></a>
### 15. [M] `normPlot` reshapes using global `nphi`/`ntheta` (Single Stage)

**File:** `single_stage_banana_example.py`
**Line:** 464

**Validation status:** Latent / design. Safe only when the evaluated field grid matches the module globals; exact-Boozer or reused surfaces can violate that assumption.

```python
def normPlot(surf, bs, filename):
    ...
    relBfinal_norm = np.sum(bs.B().reshape((nphi, ntheta, 3)) * ...)  # globals!
```

The function accepts `surf` and `bs` as parameters but reshapes the field using module-level
`nphi` and `ntheta`. If the Boozer surface has different quadrature resolution (e.g.,
`--boozer-stage final` uses `2*mpol+1` points), the reshape raises `ValueError`.

---

<a id="16"></a>
### 16. [M] `magneticFieldPlots` reshapes using global `nphi`/`ntheta` (Stage 2)

**File:** `banana_coil_solver.py`
**Line:** 317

**Validation status:** Latent / design. The current Stage 2 path typically uses matching grids, but the helper is still coupled to module globals instead of the passed surface.

```python
def magneticFieldPlots(surf, bs, OUT_DIR_ITER):
    ...
    relBfinal_norm = np.sum(bs.B().reshape((nphi, ntheta, 3)) * ...)  # globals!
```

Same issue as [Issue 15](#15). The function also redundantly recomputes `relBfinal_norm` on
line 317 after already computing it on lines 310-313, then uses `max_rnorm` from the first
computation with values from the second for the contour plot color scale.

---

<a id="17"></a>
### 17. [M] Quadrature points include duplicate endpoint

**File:** `banana_coil_solver.py`
**Line:** 182

**Validation status:** Latent / design. This is a low-grade numerical issue rather than a demonstrated current crash or wrong-answer blocker.

```python
banana_curve = CurveCWSFourierCPP(np.linspace(0, 1, num_quadpoints), order=order, surf=surf_coils)
```

`np.linspace(0, 1, 128)` includes both 0 and 1, which are the same point on a periodic curve.
For quadrature of periodic functions on [0,1), the standard is
`np.linspace(0, 1, N, endpoint=False)`. Including the endpoint double-weights the closure point,
introducing a small quadrature error in length, curvature, and flux computations.

**Fix:**
```python
np.linspace(0, 1, num_quadpoints, endpoint=False)
```

---

<a id="18"></a>
### 18. [--] ~~`BoozerResidualExact` hardcodes `constraint_weight=0`~~ (by design)

**File:** `single_stage_banana_example.py`
**Lines:** 300-301

```python
print("warning: constraint weight set to 0")
self.constraint_weight = 0.0
```

**Not a bug.** In exact Boozer mode, the inner solver (`solve_residual_equation_exactly_newton`)
enforces the volume/label constraint as a **hard equation** in the Newton system -- it appears as
a row in the Jacobian, not as a penalty. The constraint is satisfied to machine precision by the
inner solver. Setting `constraint_weight=0` in the outer `BoozerResidualExact` wrapper correctly
avoids double-penalizing an already-satisfied constraint. The misleading `print("warning: ...")`
could be clarified, but the math is correct.

---

<a id="19"></a>
### 19. [C] Exact Boozer quadpoints use `mpol` for phi -- throws for `mpol != ntor`

**File:** `single_stage_banana_example.py`
**Lines:** 417-418

```python
quadpoints_theta=np.linspace(0, 1, 2*mpol+1, endpoint=False),
quadpoints_phi=np.linspace(0, 1./surf.nfp, 2*mpol+1, endpoint=False),
```

Both dimensions use `2*mpol+1` points. The exact Boozer solver
(`solve_residual_equation_exactly_newton`) requires quadpoints matching one of three specific
patterns documented in `boozersurface.py:853-876` -- all requiring `2*ntor+1` phi points. The
`get_stellsym_mask()` method in `surfacexyztensorfourier.py:206-217` validates this and raises
`Exception('Stellarator symmetric BoozerExact surfaces require a specific set of quadrature
points on the surface...')` when the pattern does not match.

With defaults `mpol=8, ntor=6`: the code creates 17 phi points (`2*8+1`) instead of the
required 13 (`2*6+1`). This **throws an exception** during `boozer_surface.run_code()`, blocking
the `--boozer-stage final` path **before** [Issue 1](#1) (missing imports) is even reached. For
`mpol == ntor` (e.g., both 8), the counts coincidentally match and no error occurs.

This is the **first** failure point in the exact-Boozer code path: `#19` throws, then (if fixed)
`#1` crashes.

**Fix:**
```python
quadpoints_phi=np.linspace(0, 1./surf.nfp, 2*ntor+1, endpoint=False),
```

**Candidate update (2026-03-16):** Fixed locally in the candidate working tree by changing the
exact Boozer `phi` quadrature to `2*ntor+1`. Validated on the normal
`--boozer-stage final --mpol 8 --ntor 6 --init-only` candidate path, which no longer failed
immediately at the old `mpol != ntor` exact-surface construction error, and covered by a
dedicated regression test in
`/Users/suhjungdae/code/columbia/simsopt/tests/geo/test_single_stage_example.py`. This note
applies to the local candidate only, not the historical baseline.

---

<a id="20"></a>
### 20. [--] ~~`callback` re-calls `JF.J()` / `JF.dJ()`~~ (cache-safe)

**File:** `single_stage_banana_example.py`
**Lines:** 594-595

**Validation status:** Not a bug. The rereads are warm-cache and do not materially repeat the expensive Boozer solve in the current call pattern.

```python
run_dict['J'] = JF.J()
run_dict['dJ'] = JF.dJ().copy()
```

The callback is invoked after a successful L-BFGS-B iteration. `fun()` already computed `J` and
`dJ` during the last function evaluation. SIMSOPT's `Optimizable` base class only invalidates
caches on new DOFs (`set_recompute_flag` in `optimizable.py`), and `BoozerSurface.run_code()`
returns early when `need_to_run_code` is `False` (`boozersurface.py:175`). Since DOFs have not
changed between `fun()` and `callback()`, these calls hit warm caches and are inexpensive. Not a
performance bug; the earlier claim of "approximately doubles the cost" was incorrect. Caching
from `fun()` would be slightly cleaner but is not necessary.

---

<a id="21"></a>
### 21. [M] `bs.B()` reshape assumes nphi*ntheta before `set_points`

**File:** `single_stage_banana_example.py`
**Lines:** 769-770

**Validation status:** Latent / design. The default initial-stage path usually succeeds because `run_code()` happened to populate matching points, but the coupling is implicit and fragile.

```python
pointData = {"B_N/B": np.sum(bs.B().reshape((nphi, ntheta, 3)) *
    boozer_surface.surface.unitnormal(), axis=2)[:, :, None] / ...}
```

At this point in the code, `bs.set_points(...)` has not been called on
`boozer_surface.surface.gamma()`. The field `bs` was loaded from file (line 730) and may have
been last evaluated at a different grid. If the Boozer surface has different quadrature
resolution than `nphi * ntheta` (e.g., `--boozer-stage final`), this reshape will fail.

In the default `--boozer-stage initial` path, the Boozer surface inherits quadpoints from
`surf_prev` (which uses `nphi * ntheta`), and `initialize_boozer_surface` calls
`boozer_surface.run_code()` which internally calls `bs.set_points(...)`, so the reshape
succeeds. But this coupling is implicit and fragile.

---

<a id="22"></a>
### 22. [M] Hardcoded relative `OUT_DIR` in Poincare script

**File:** `poincare_surfaces.py`
**Line:** 94

```python
OUT_DIR = f'../SINGLE_STAGE/outputs/mpol=8-ntor=6'
```

Only works when run from the `POINCARE_PLOTTING/` directory. Unlike the other scripts, there is
no CLI argument or environment variable to override it.

---

<a id="23"></a>
### 23. [M] Duplicated plotting code with identical bugs across files

**Files:** `banana_coil_solver.py` and `single_stage_banana_example.py`

**Validation status:** Latent / design. This is a maintainability problem that amplifies bugs and increases fix cost rather than a standalone runtime failure.

`crossSectionPlot` and `normPlot` (along with `magneticFieldPlots`) are copy-pasted between
the two scripts with near-identical code and the same bugs in each:

- Both `crossSectionPlot` copies have the toroidal angle double-scaling ([#8](#8), [#9](#9))
- Both `crossSectionPlot` copies have dead `hbt_poly` ([#25](#25), [#26](#26))
- Both `crossSectionPlot` copies depend on globals `hbt` and `VV`
- Both `normPlot`/`magneticFieldPlots` copies reshape using global `nphi`/`ntheta`

Every bug fix must be applied twice.

---

<a id="24"></a>
### 24. [M] No `if __name__ == "__main__"` guard in any script

**Files:** All three Python scripts

**Validation status:** Latent / design. The scripts behave as expected when executed directly, but importing them for tests or orchestration immediately runs the workflow.

- `banana_coil_solver.py` begins executing at line 393
- `single_stage_banana_example.py` begins executing at line 672
- `poincare_surfaces.py` begins executing at line 84

Importing any of these modules (for testing, orchestration, or reuse) will immediately trigger
the full optimization workflow and will be brittle to global initialization order.

---

## Low Issues

<a id="25"></a>
### 25. [L] Dead `hbt_poly` variable (Stage 2)

**File:** `banana_coil_solver.py`
**Line:** 354

```python
hbt_poly = Polygon(zip(rs3, zs3))  # created but never used
```

The `shapely.geometry.Polygon` import (line 18) exists solely for this dead variable.

---

<a id="26"></a>
### 26. [L] Dead `hbt_poly` variable (Single Stage)

**File:** `single_stage_banana_example.py`
**Line:** 487

```python
hbt_poly = Polygon(zip(rs3, zs3))  # created but never used
```

The `shapely.geometry.Polygon` import (line 6) exists solely for this dead variable.

---

<a id="27"></a>
### 27. [L] Unused imports (Stage 2)

**File:** `banana_coil_solver.py`
**Lines:** 13-22

The following imports are not used anywhere in the file:

| Line | Import |
|------|--------|
| 13 | `InterpolatedField` |
| 14-15 | `SurfaceClassifier`, `compute_fieldlines`, `LevelsetStoppingCriterion`, `plot_poincare_data` |
| 18 | `Polygon` (only used for dead `hbt_poly`) |
| 19 | `copy` |
| 20 | `shutil` |
| 22 | `combinations` |

---

<a id="28"></a>
### 28. [L] Unused imports (Single Stage)

**File:** `single_stage_banana_example.py`

| Line | Import | Notes |
|------|--------|-------|
| 6 | `Polygon` | Only used for dead `hbt_poly` |
| 16 | `save` | Imported from `simsopt._core.optimizable` but never called |
| 17 | `ScaledCurrent` | Imported from `simsopt.field.coil` but never used (coils are loaded from JSON) |

---

<a id="29"></a>
### 29. [L] Unused imports (Poincare)

**File:** `poincare_surfaces.py`

| Line | Import | Notes |
|------|--------|-------|
| 4-5 | `SurfaceClassifier`, `LevelsetStoppingCriterion` | Imported but never referenced |
| 8 | `SurfaceRZFourier` | Imported but never used (surfaces are loaded from JSON) |
| 9 | `Line2D` | Imported but never used |

---

<a id="30"></a>
### 30. [L] `tol=1e-15` is exceptionally strict (Stage 2)

**File:** `banana_coil_solver.py`
**Line:** 509

**Validation status:** Latent / tuning. This is not a correctness bug, but the original
tolerance was implicit (via the blunt `tol=` parameter) and undocumented.

```python
# Original:
res = minimize(fun, dofs, jac=True, method='L-BFGS-B',
               options={'maxiter': MAXITER, 'maxcor': 300}, tol=1e-15)
# Fixed — explicit ftol/gtol in options, configurable via --ftol/--gtol CLI flags:
res = minimize(fun, dofs, jac=True, method='L-BFGS-B',
               options={'maxiter': MAXITER, 'maxcor': 300, 'ftol': args.ftol, 'gtol': args.gtol})
```

SciPy maps `tol` to both `ftol` and `gtol` for L-BFGS-B via `options.setdefault()`. The
original `tol=1e-15` set both to `1e-15`:
- `ftol=1e-15` → `factr ≈ 4.5` (relative function change within ~4.5 machine epsilons)
- `gtol=1e-15` → projected gradient norm below 1e-15

These thresholds are exceptionally strict and likely impractical for this optimization problem,
though not unreachable in principle (projected-gradient termination can trigger at machine
precision for well-converged problems). In practice, the optimizer will most likely terminate
by hitting `maxiter`.

**Fix (Stage 2 only):** Replaced the implicit `tol=1e-15` with explicit `--ftol`/`--gtol` CLI
flags (defaults: `1e-15` each, preserving the original very-tight termination settings). The
tolerances are now set directly in the `options` dict, bypassing the `setdefault` indirection.
Users can loosen for faster convergence (e.g., `--ftol 1e-9 --gtol 1e-5` for scipy "moderate
accuracy" defaults). Note: Single stage already had its own explicit `ftol_by_mpol`/`gtol_by_mpol`
system (Issue #31).

Additionally, `maxcor=300` stores 300 correction pairs (default is 10). With typical ~20-50
DOFs this is fine memory-wise but far exceeds what is useful for the Hessian approximation.

---

<a id="31"></a>
### 31. [L] `ftol`/`gtol` returns `None` for out-of-range `mpol`

**File:** `single_stage_banana_example.py`
**Lines:** 852-853

```python
ftol = ftol_by_mpol.get(mpol)   # None if mpol not in {8..18}
gtol = gtol_by_mpol.get(mpol)   # None if mpol not in {8..18}
```

The dictionaries cover `mpol` 8-18. For `mpol < 8` or `mpol > 18`, `ftol` and `gtol` are
`None`. SciPy's L-BFGS-B wrapper (`_lbfgsb_py.py`) divides `ftol` directly without a None
guard, so passing `ftol=None` raises `TypeError` at runtime. This is a crash, not a silent
fallback.

**Fix:** Add a default, e.g. `ftol_by_mpol.get(mpol, 1e-5)`.

**Candidate update (2026-03-16):** Fixed locally with edge-aware defaults:
`ftol_by_mpol.get(mpol, 1e-5 if mpol < 8 else 1e-10)` and
`gtol_by_mpol.get(mpol, 1e-2 if mpol < 8 else 1e-7)`. For mpol < 8, uses mpol=8 tolerances
(most relaxed); for mpol > 18, uses mpol=18 tolerances (most strict). Confirmed scipy
`_lbfgsb_py.py` divides `ftol` at line 376 (`factr = ftol / np.finfo(float).eps`) — `None`
causes `TypeError`.

---

<a id="32"></a>
### 32. [L] Leaked matplotlib figure (Poincare)

**File:** `poincare_surfaces.py`
**Lines:** 30-31

```python
plt.figure()                                          # creates figure 1 (leaked)
fig, axs = plt.subplots(nrowcol, nrowcol, ...)        # creates figure 2 (used)
```

`plt.figure()` creates a figure that is immediately superseded by `plt.subplots()`. The first
figure is never closed.

---

<a id="33"></a>
### 33. [L] Y-axis tick labels only hidden for column 1 (Poincare)

**File:** `poincare_surfaces.py`
**Line:** 47

**Validation status:** Conditional. Not triggered by the current script because `phis` is hardcoded to 4, yielding a 2x2 grid; becomes real once the plot has more than 2 columns.

```python
if col == 1:
    axs[row, col].set_yticklabels([])
```

For `nrowcol > 2`, columns 2, 3, etc. still show redundant y-axis tick labels.

**Fix:** `if col > 0:`

---

<a id="34"></a>
### 34. [L] Blank subplots for non-square phi count (Poincare)

**File:** `poincare_surfaces.py`
**Line:** 29

**Validation status:** Conditional. Not triggered by the current script because `phis` is hardcoded to 4; becomes visible for non-square phi counts.

```python
nrowcol = ceil(sqrt(len(phis)))
```

For 3 phis, `nrowcol=2` creates 4 subplots with one blank. Unused axes are not hidden. Should
call `ax.set_visible(False)` for unused subplot positions.

---

<a id="35"></a>
### 35. [L] Shell scripts hardcoded to `~/simsopt/` and `conda activate simsopt`

**Files:** `banana-scan.sh:31`, `single-scan.sh:32`, `poincare-plot.sh:30`

```bash
cd ~/simsopt/examples/single_stage_optimization/STAGE_2      # banana-scan.sh
cd ~/simsopt/examples/single_stage_optimization/SINGLE_STAGE  # single-scan.sh
cd ~/simsopt/examples/single_stage_optimization/POINCARE_PLOTTING  # poincare-plot.sh
```

All three scripts assume the repo lives at `~/simsopt/` and that a conda environment named
`simsopt` exists. Not portable to other cluster accounts or installation layouts.

**Fix:** Use `$SLURM_SUBMIT_DIR` or a configurable variable for the working directory.

---

<a id="36"></a>
### 36. [L] `fun()` closes over many globals (Stage 2)

**File:** `banana_coil_solver.py`
**Lines:** 377-388

**Validation status:** Latent / design. Functional today, but tightly coupled and harder to test or reuse safely.

```python
def fun(dofs):
    JF.x = dofs
    ...
    BdotN = np.mean(np.abs(np.sum(new_bs.B().reshape((nphi, ntheta, 3)) * new_surf.unitnormal(), axis=2)))
```

`JF`, `new_bs`, `new_surf`, `nphi`, `ntheta`, `Jf`, `Jls`, `Jccdist`, `Jc` are all captured
from module scope. Functional, but fragile and hard to test in isolation.

---

<a id="37"></a>
### 37. [--] ~~Default `s=0.24` is an interior VMEC surface~~ (intentional)

**File:** `banana_coil_solver.py`
**Line:** 81

```python
default=float(os.environ.get("TOROIDAL_FLUX", "0.24")),
```

VMEC flux label `s=0.24` corresponds to ~49% of the minor radius (`r ~ sqrt(s)`). While Stage 2
coil optimization more commonly targets the LCFS (`s=1.0`), this default appears intentional for
the HBT-hybrid workflow, as the same value is used consistently in the default seed table
(`DEFAULT_STAGE2_SEEDS_BY_PLASMA`) and across both Stage 2 and single-stage scripts. Not a bug,
but the choice could benefit from a code comment explaining the rationale.

---

## Validation Log

This issue tracker was reviewed against the SIMSOPT source and SciPy 1.17.1.

**Round 1 corrections (2026-03-14):**

| # | Original | Corrected | Reason |
|---|----------|-----------|--------|
| 2 | "corrupts Hessian" | "can corrupt Hessian" | L-BFGS-B curvature check may skip the update |
| 6 | long→parallel, short→missed | short→false parallel, long→missed | `denom = a*c*sin^2(θ)` scale analysis was reversed |
| 18 | M (bug) | -- (by design) | Exact solver enforces label as hard constraint; outer weight=0 avoids double penalty |
| 19 | M (underresolved) | C (throws) | `get_stellsym_mask()` raises Exception for `mpol != ntor`; first failure in exact path |
| 20 | M (doubles cost) | L (may partially recompute) | SIMSOPT caches via `new_x` flag; most expensive ops not repeated |
| 28 | 2 unused imports | 3 unused imports | `ScaledCurrent` (line 17) also unused |
| 30 | "tol is ignored" | "tol maps correctly but is virtually unreachable" | SciPy maps `tol` to `ftol`+`gtol` via `setdefault` |
| 37 | L (possible bug) | -- (intentional) | Consistent with seed tables; workflow-specific choice |

**Round 2 corrections (2026-03-14):**

| # | Original | Corrected | Reason |
|---|----------|-----------|--------|
| 20 | L (may partially recompute) | L (cache-safe, not a perf bug) | `BoozerSurface.run_code()` returns early; `new_x` not set between `fun()` and `callback()` |
| 30 | "virtually unreachable" | "exceptionally strict, likely impractical" | Runtime test showed projected-gradient termination can trigger at 1e-15 |
| 31 | "L-BFGS-B uses defaults for None" | "SciPy raises TypeError for None" | `_lbfgsb_py.py` divides `ftol` directly; no None guard |

**Round 3 validation pass (2026-03-15):**

| # | Updated classification | Reason |
|---|------------------------|--------|
| 13 | latent / design | Depends on globals but works in the current script order |
| 14 | latent / design | Misleading function signature; placement still uses globals |
| 15 | latent / design | Real only when field-grid shape diverges from module globals |
| 16 | latent / design | Same as #15 in Stage 2; helper is globally coupled |
| 17 | latent / design | Small periodic-quadrature issue, not a demonstrated runtime blocker |
| 20 | -- (not a bug) | Cache-safe rereads; warm-cache path does not duplicate the expensive solve |
| 21 | latent / design | Current path relies on implicit prior `set_points()` side effect |
| 23 | latent / design | Maintainability issue, not an independent runtime defect |
| 24 | latent / design | Import-time execution is fragile for tests/reuse, but scripts still run directly |
| 30 | latent / tuning | Strict optimizer tolerance choice, not a correctness failure |
| 33 | conditional | Current hardcoded 4-phi plot gives only 2 columns, so the bug is dormant |
| 34 | conditional | Current hardcoded 4-phi plot yields a full 2x2 grid, so no blank panel appears |
| 36 | latent / design | Global captures are functional but tightly coupled and hard to test |

**Round 4 deep-dive (2026-03-16):**

| # | Original claim | Corrected | Reason |
|---|----------------|-----------|--------|
| 2 | "can corrupt Hessian" (mechanism: negative curvature y_k poisons BFGS) | TRUE in impact, mechanism refined | Empirical 20D Rosenbrock tests against scipy 1.17.1 show: (a) transient failures are harmless — line search rejects fake `(J_old, -dJ_old)` because Armijo fails, backtracks, next eval succeeds; (b) persistent failures cause `ABNORMAL_TERMINATION_IN_LNSRCH` with BFGS memory **flushed** (reset to identity), not corrupted with bad curvature pairs — the line search rejects the step before the BFGS update is reached. Fix: `(J_old + penalty, dJ_old)` — elevated J triggers backtracking; same-sign gradient produces `y_k=0` if step is ever accepted via WARN path, safely skipping the BFGS update. Regression test added. |
| 2 | Suggested fix: `(J + LARGE_PENALTY, zero_gradient)` | Not recommended | Zero gradient produces `y_k = -dJ_old` which passes the curvature check `y^T s > 0` for descent directions; oversized penalty can dominate the landscape (empirically converged to wrong point in testing). |

---

## Statistics

- **Total entries:** 37
- **Active / cleanup issues:** 22
- **Latent / design / conditional issues:** 12
- **Reclassified as not bugs:** 3
- **Critical:** 8
- **Moderate:** 14
- **Low:** 12
- **Not bugs:** 3

### By file

| File | C | M | L | -- | Total |
|------|---|---|---|----|-------|
| `banana_coil_solver.py` | 2 | 7 | 4 | 1 | 14 |
| `single_stage_banana_example.py` | 5 | 4 | 3 | 2 | 14 |
| `poincare_surfaces.py` | 0 | 1 | 4 | 0 | 5 |
| `poincare-plot.sh` | 1 | 0 | 0 | 0 | 1 |
| Cross-file / shell | 0 | 2 | 1 | 0 | 3 |

### By category

| Category | Count |
|----------|-------|
| Math / Geometry | 4 |
| Physics / API | 5 |
| Python / Scoping / Runtime | 4 |
| Optimization Correctness | 3 |
| Reproducibility / Provenance | 2 |
| Software Engineering / Design | 10 |
| Dead Code / Unused Imports | 5 |
| Shell / Portability | 2 |
| Reclassified / not bugs | 3 |
