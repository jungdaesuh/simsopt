# Performance, Memory & Algorithm Efficiency Issues

Companion to `ISSUES.md` (correctness/design audit). This document covers efficiency
dimensions not addressed by the 37+1 correctness issues.

Review date: 2026-03-17. Analyzed against the post-Batch-6 candidate code.

**Erratum (2026-03-17):** Corrections applied after reviewer deep-dive validated findings
P1–P3 against the actual candidate code and SciPy internals. See individual items for
details.

---

## Priority Legend

| Tag | Meaning |
|-----|---------|
| **H** | High — measurable wallclock or memory improvement |
| **L** | Low — negligible improvement, cosmetic or theoretical |

---

## Table of Contents

| # | Pri | Category | File | Short Title |
|---|-----|----------|------|-------------|
| [N1](#n1) | H | Perf / Memory | Stage 2 `fun()` | BdotN diagnostic in hot loop (~1 GiB cumulative throwaway) |
| [N2](#n2) | L | Memory | Single Stage `callback()` | 8 gradient arrays allocated for scalar norms |
| [N5](#n5) | H | Perf | Both `fun()` | Diagnostics run during line-search probes |
| [N6](#n6) | L | Perf | Single Stage `fun()` | `Iotas()` object constructed every eval |
| [N7](#n7) | H | Perf | Single Stage `fun()` | `is_self_intersecting()` in every eval including line-search |
| [N8](#n8) | H | Algo | Stage 2 `check_all_pairs` | Symmetric pairs checked twice |
| [N9](#n9) | L | Algo | Stage 2 `check_all_pairs` | O(n²) brute-force without spatial indexing |
| [N10](#n10) | L | Algo | Both `minimize()` | `maxcor=300` for ~11 DOFs |
| [N11](#n11) | L | Thread | Stage 2 `check_all_pairs` | No `@njit(parallel=True)` with `prange` |
| [N12](#n12) | L | Immutable | Single Stage `dJ_by_dB` | In-place `/=` mutates arrays from external function |

---

## Issues

<a id="n1"></a>
### N1. [H] BdotN diagnostic in optimizer hot loop

**File:** `banana_coil_solver.py`, `fun()`
**Category:** Performance / Memory

```python
def fun(dofs):
    JF.x = dofs
    J = JF.J()
    grad = JF.dJ()
    # --- Everything below is diagnostic logging ---
    BdotN = np.mean(np.abs(np.sum(new_bs.B().reshape((nphi, ntheta, 3)) * new_surf.unitnormal(), axis=2)))
    outstr += f", Len={Jls.J():.1f}m"
    outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f}m"
    ...
    print(outstr)
    return J, grad
```

The BdotN computation allocates temporary arrays at `255×64×3 = 48,960` float64 elements
(~0.374 MiB each). The exact temporary count depends on which intermediate expressions
produce views vs copies — `.reshape()` is typically a view, but the element-wise multiply
and `np.sum` produce new arrays. A conservative estimate is 2–3 materialized temporaries
per call (~0.75–1.1 MiB). L-BFGS-B calls `fun()` multiple times per iteration
(line-search), so this runs ~3–10x per accepted step. Over 300 iterations with ~5 probes
each: ~1500 calls, ~1200 of which produce diagnostic output nobody reads (only the
callback-iteration output matters). Cumulative throwaway allocation: ~0.9–1.3 GiB for the
temporaries alone, plus the `B()` evaluation cost.

**Estimated savings:** ~2–5% wallclock; ~1 GiB cumulative allocation eliminated. Bigger win
is cleaner stdout — currently produces ~5 diagnostic lines per iteration instead of 1.

**Fix:** Move all diagnostic computation (lines after `grad = JF.dJ()`) into `callback()`,
which runs once per accepted step.

---

<a id="n2"></a>
### N2. [L] 8 gradient arrays allocated for scalar norms in callback

**File:** `single_stage_banana_example.py`, `callback()`

```python
dJ_QS = np.linalg.norm(JnonQSRatio.dJ())
dJ_Boozer = np.linalg.norm(JBoozerResidual.dJ())
dJ_iota = np.linalg.norm(Jiota.dJ())
dJ_len = np.linalg.norm(JCurveLength.dJ())
dJ_cc = np.linalg.norm(JCurveCurve.dJ())
dJ_cs = np.linalg.norm(JCurveSurface.dJ())
dJ_surf = np.linalg.norm(JSurfSurf.dJ())
dJ_curvature = np.linalg.norm(JCurvature.dJ())
```

Each `.dJ()` returns a full gradient array (size = DOF count), allocated only to compute a
scalar norm, then immediately discarded. Runs once per accepted step.

**Estimated savings:** Negligible — microseconds per iteration.

**Fix:** Not worth the code churn. The allocations are small and the callback runs infrequently.

---

<a id="n5"></a>
### N5. [H] Diagnostics run during line-search probes (both scripts)

**Files:** `banana_coil_solver.py` `fun()`, `single_stage_banana_example.py` `fun()`
**Category:** Performance

Same root cause as [N1](#n1). `fun()` is called by L-BFGS-B's line search (Wolfe conditions),
which probes ~3–10 points per iteration. Only the final accepted point matters for
diagnostics. All intermediate probes print diagnostics that are noise.

In single-stage `fun()`, the overhead per probe includes:
- `Iotas(boozer_surface).J()` object construction + evaluation ([N6](#n6))
- `boozer_surface.surface.is_self_intersecting()` O(n²) check ([N7](#n7))
- `print()` I/O flush

**Fix (Stage 2 only):** In `banana_coil_solver.py`, the diagnostic lines after
`grad = JF.dJ()` can be moved directly to `callback()` — Stage 2's `fun()` is a simple
`JF.x = dofs; J = JF.J(); grad = JF.dJ()` wrapper.

**Fix (Single Stage — NOT the same):** In `single_stage_banana_example.py`, `fun()` cannot
be reduced to `JF.x = dofs; return JF.J(), JF.dJ()`. It must first reset the Boozer surface
to the last accepted state, run `boozer_surface.run_code()`, validate success, check
self-intersection, and apply the fallback rollback path before `JF.J()` / `JF.dJ()` are
meaningful. The diagnostic `print()` statements and `Iotas()` construction ([N6](#n6)) on the
success path can be moved to `callback()`, but the Boozer solve and validation logic must
remain in `fun()`.

---

<a id="n6"></a>
### N6. [L] `Iotas()` object constructed every `fun()` eval

**File:** `single_stage_banana_example.py`, `fun()`

```python
print(f"Iota: {Iotas(boozer_surface).J()}")
```

Constructs a new `Iotas` object (which calls `Optimizable.__init__` and sets up dependency
tracking) every `fun()` evaluation, then discards it. The `iota` object already exists at
module scope.

**Estimated savings:** Negligible — object construction is ~microseconds vs seconds for the
Boozer solve.

**Fix:** Replace with `iota.J()` (module-scope object). One-line change.

---

<a id="n7"></a>
### N7. [H] `is_self_intersecting()` in every `fun()` eval including line-search

**File:** `single_stage_banana_example.py`, `fun()`

```python
success2 = not boozer_surface.surface.is_self_intersecting()
```

`SurfaceXYZTensorFourier.is_self_intersecting()` is an O(n²) check in the number of surface
quadrature points. This runs in **every** `fun()` evaluation, including L-BFGS-B line-search
probes that will be discarded.

**Estimated savings:** Potentially 5–15% wallclock depending on surface resolution. This is
the single highest-value performance fix.

**Tradeoff:** The self-intersection check is part of single-stage's `fun()` validation
logic — it gates whether `JF.J()` / `JF.dJ()` are returned or the fallback penalty path is
taken. It **cannot** be moved to `callback()` as a post-hoc rollback. SciPy's L-BFGS-B
invokes `callback()` only after a new iterate is accepted (`task[0] == 1` in the Fortran
driver), at which point the optimizer has already advanced its internal state (`x`, `f`,
BFGS memory). A callback that detects self-intersection cannot undo the accepted step.

**Safe alternatives:**
- **Amortize:** Check every N `fun()` evaluations instead of every one. On intervening
  calls, skip the check and accept the Boozer solve success alone. If the next check finds
  self-intersection, the fallback path triggers. Risk: up to N-1 function evaluations may
  use a self-intersecting surface — acceptable if self-intersection develops gradually
  rather than appearing suddenly.
- **Cache with dirty flag:** Only recheck when `boozer_surface.surface.x` has changed since
  the last check. If the line search probes nearby points that don't change the surface
  topology, the cached result is reused.

---

<a id="n8"></a>
### N8. [H] Symmetric pairs checked twice in `check_all_pairs`

**File:** `banana_coil_solver.py`, `check_all_pairs()`

```python
for i in range(n_segments):
    for j in range(n_segments):
```

Segment distance is symmetric: `dist(i,j) == dist(j,i)`. The loop checks each pair twice.

With `npts=2000`: 4,000,000 iterations → 2,000,000 with the fix.

**Estimated savings:** 2x speedup for `is_self_intersecting()`. For Stage 2 (called once
post-optimization), this saves ~0.5 seconds. For Single Stage where [N7](#n7) puts it in the
hot loop, the savings compound.

**Fix:**
```python
for i in range(n_segments):
    for j in range(i + 1, n_segments):
```
Plus adjust neighbor-skip logic for the one-directional iteration.

---

<a id="n9"></a>
### N9. [L] O(n²) self-intersection without spatial indexing

**File:** `banana_coil_solver.py`, `check_all_pairs()`

Brute-force all-pairs check with `npts=2000` does ~4M (or ~2M after [N8](#n8)) segment-pair
distance computations. An axis-aligned bounding box (AABB) sweep-and-prune would reduce to
O(n log n) comparisons with O(n) candidate pairs — roughly 180x fewer distance computations.

**Estimated savings:** ~100–200x for `is_self_intersecting()` at n=2000. But in Stage 2 this
function is called once post-optimization (<1 second). Only matters in Single Stage if
[N7](#n7) keeps it in the hot loop.

**Fix:** Implement sweep-and-prune or a simple spatial hash grid. Only worth doing if N7
remains (self-intersection checked per `fun()` eval).

---

<a id="n10"></a>
### N10. [L] `maxcor=300` for ~11 DOFs

**Files:** `banana_coil_solver.py`, `single_stage_banana_example.py`

```python
options={'maxiter': MAXITER, 'maxcor': 300}
```

L-BFGS-B stores `maxcor` correction vector pairs to approximate the Hessian inverse.
Research shows diminishing returns beyond `maxcor ≈ 2n` where `n` is the DOF count. The
current HBT banana-coil configuration has ~11 active DOFs (`bs.x.size == 11` in the
candidate Stage 2 output). `maxcor=300` is ~27x the DOF count; `maxcor=20–30` would be
sufficient.

**Estimated savings:** Negligible — the Hessian update is ~microseconds per iteration.
Memory savings: trivial at 11 DOFs.

**Fix:** Not worth changing. The cost is dominated by `JF.J()` / `JF.dJ()`, not the BFGS
update. The conclusion holds even more strongly at 11 DOFs than at the previously stated
20–50.

---

<a id="n11"></a>
### N11. [L] `check_all_pairs` doesn't use `@njit(parallel=True)`

**File:** `banana_coil_solver.py`, `check_all_pairs()`

```python
@njit
def check_all_pairs(segments, tol, neighbor_skip):
    for i in range(n_segments):
        for j in range(n_segments):
```

The SLURM scripts allocate 16 CPUs. Using `@njit(parallel=True)` with `prange` on the outer
loop would parallelize across all cores.

**Estimated savings:** ~10–16x speedup for `check_all_pairs`. Only matters for Stage 2
post-optimization call (where it saves <1 second) or in Single Stage if [N7](#n7) keeps the
check in the hot loop.

**Fix:**
```python
@njit(parallel=True)
def check_all_pairs(segments, tol, neighbor_skip):
    for i in prange(n_segments):
```
Note: early-return (`return True`) inside `prange` requires a shared flag pattern since
`prange` doesn't support early termination.

---

<a id="n12"></a>
### N12. [L] In-place `/=` mutates arrays from external function

**File:** `single_stage_banana_example.py`, `BoozerResidualExact.dJ_by_dB()`

```python
r, r_dB = boozer_surface_residual_dB(surface, ...)
r /= np.sqrt(num_points)      # in-place mutation
r_dB /= np.sqrt(num_points)   # in-place mutation
```

`r` and `r_dB` are returned by `boozer_surface_residual_dB()`. In-place `/=` mutates these
arrays. If any other code holds a reference to the same backing memory, it sees corrupted
values.

**Estimated savings:** Zero performance benefit. Pure safety.

**Risk:** Currently safe because nobody else holds a reference. Fragile under refactor.

**Fix:**
```python
r = r / np.sqrt(num_points)
r_dB = r_dB / np.sqrt(num_points)
```

---

## Priority Matrix

```
                    Low effort          High effort
                ┌───────────────────┬───────────────────┐
  High value    │ N8 (half pairs)   │ N7 (amortize or   │
                │ N6 (reuse iota)   │   cache check)    │
                │ N1+N5 (move diag  │                   │
                │   to callback)    │                   │
                ├───────────────────┼───────────────────┤
  Low value     │ N12 (copy vs /=)  │ N9 (AABB index)   │
                │                   │ N11 (prange)       │
                │                   │ N10 (maxcor)       │
                │                   │ N2 (grad norms)    │
                └───────────────────┴───────────────────┘
```

**Recommended fix order:**
1. **N1+N5** — move diagnostic prints from `fun()` to `callback()`. Stage 2: straightforward.
   Single Stage: move only the `print()` statements and `Iotas()` construction; Boozer solve
   and validation must stay in `fun()`.
2. **N8** — `range(i+1, n_segments)` (one-line change, 2x on self-intersection)
3. **N6** — `iota.J()` instead of `Iotas(boozer_surface).J()` (one-line change)
4. **N7** — amortize or cache `is_self_intersecting()` (medium effort, biggest win; callback
   rollback is NOT viable — see N7 tradeoff section)
5. Everything else: skip unless profiling shows a bottleneck.
