# Parity Audit 09: Wireframe Optimization Kernels (RCLS, GSCO)

## Header

- **Audit timestamp:** 2026-05-16
- **Files audited:**
  - JAX: `src/simsopt/solve/wireframe_optimization_jax.py` (875 lines)
  - C++:  `src/simsoptpp/wireframe_optimization.cpp` (538 lines)
  - C++ header: `src/simsoptpp/wireframe_optimization.h` (31 lines)
  - CPU orchestrator/RCLS reference: `src/simsopt/solve/wireframe_optimization.py` (859 lines)
  - JAX parity test (Item 31): `tests/solve/test_wireframe_optimization_jax_item31.py` (725 lines)
  - Legacy CPU test: `tests/solve/test_wf_optimization.py` (530 lines)
- **Algorithms in scope:**
  - **RCLS** (Regularized Constrained Least Squares) — implemented in pure Python (NumPy/SciPy) in `wireframe_optimization.py::regularized_constrained_least_squares`, ported in `wireframe_optimization_jax.py::regularized_constrained_least_squares_jax`. **No C++ implementation exists** for RCLS.
  - **GSCO** (Greedy Stellarator Coil Optimization) — C++ kernel at `wireframe_optimization.cpp::GSCO` (lines 72-349), JAX kernel at `wireframe_optimization_jax.py::greedy_stellarator_coil_optimization_jax` (lines 296-516).

## Executive Summary

Top three findings, ordered by impact:

1. **[HIGH] GSCO `stop_undone_loop` test is direction-asymmetric and matches a C++ operator-precedence quirk by design.** Both implementations only detect undo against the **positive→negative** direction (`opt_ind + nLoops == opt_ind_prev`), not the **negative→positive** direction (`opt_ind - nLoops == opt_ind_prev`). This is a latent C++ bug at `wireframe_optimization.cpp:270` (`(opt_ind + nLoops % (twoNLoops))` parses as `opt_ind + (nLoops % twoNLoops)` due to `%`-binding tighter than `+`), and the JAX implementation faithfully mirrors it at `wireframe_optimization_jax.py:406`. Parity is preserved but the algorithm under both backends will continue past a `negative→positive` undo flip and only catch it on the next subsequent `positive→negative` flip, potentially overshooting the minimum.

2. **[MEDIUM] GSCO f_B floating-point reduction order diverges between C++ and JAX, but the parity test uses `direct_kernel` tolerances (`rtol=1e-10`).** C++ accumulates `two_f_B` per candidate by a serial inner loop over grid points (`wireframe_optimization.cpp:225-235`), summing into a scalar `two_f_B_pos`. The JAX path forms a dense `nGrid × 2*nLoops` field-delta matrix and reduces via `jnp.sum((residual[:, None] + field_delta) ** 2, axis=0)` (`wireframe_optimization_jax.py:280`). When `argmin` is followed by an `accept_loop`, JAX also reuses the cached `two_f_bs[opt_ind]` rather than recomputing `residual + delta` from scratch — meaning the **same-iteration f_B history can differ from a fresh sum-of-squares by 1 ulp drift**, but it is consumed deterministically. The current parity gate (lines 32-34 of the Item 31 test) holds, but this is not a strict byte-identity guarantee on different reduction shapes (CUDA/grouped reductions).

3. **[MEDIUM/INFO] RCLS port uses `jnp.linalg.solve(R.T, d)` instead of `scipy.linalg.solve_triangular(R.T, d, lower=True)`, and `jnp.linalg.lstsq(LHS, RHS, rcond=None)` instead of `scipy.linalg.lstsq(LHS, RHS)[0]` (default `cond`).** The numerical solvers differ in default cutoffs and small-singular-value handling. Tests pass at `rtol=1e-10` for the seeded fixtures, but on degenerate/near-singular `LHS` (e.g., severely under-determined wireframes), the JAX path may select a different minimum-norm solution than SciPy's gelsd/gelss. There is no test that probes this regime.

The next subsections itemize per-function parity.

## Inventory and Function-by-Function Parity Matrix

### Module-level entry points

| Concern | C++ symbol | CPU/NumPy ref | JAX port | Parity |
|---|---|---|---|---|
| RCLS core solver | (none) | `wireframe_optimization.py:723 regularized_constrained_least_squares` | `wireframe_optimization_jax.py:102 regularized_constrained_least_squares_jax` | math equivalent; solver primitives differ (see Subsection a) |
| RCLS adapter | (none) | `wireframe_optimization.py:456 rcls_wireframe` | `wireframe_optimization_jax.py:574 rcls_wireframe_jax` | functional + immutable; constraint matrix sourced from host `wframe` |
| GSCO kernel | `wireframe_optimization.cpp:72 GSCO` | (delegates to C++) | `wireframe_optimization_jax.py:296 greedy_stellarator_coil_optimization_jax` | Faithful port of math/eligibility/history; see Subsection b |
| GSCO eligibility | `wireframe_optimization.cpp:393 check_eligibility` | (delegates to C++) | `wireframe_optimization_jax.py:182 _gsco_candidate_currents` | Matches all eligibility branches; see Subsection b |
| GSCO objective sweep | `wireframe_optimization.cpp:210-256 (parallel for)` | (delegates to C++) | `wireframe_optimization_jax.py:263 _gsco_candidate_objectives` | Vectorized via dense `A[:, loop_inds]` indexing; see Subsection b |
| GSCO active-segment count | `wireframe_optimization.cpp:371 compute_f_S` | (delegates to C++) | `wireframe_optimization_jax.py:178 _gsco_two_f_s` | Match |
| Public orchestrator | (none) | `wireframe_optimization.py:18 optimize_wireframe` | `wireframe_optimization_jax.py:729 optimize_wireframe_jax` | match in behavior |
| GSCO iteration replay | (none) | `wireframe_optimization.py:677 get_gsco_iteration` | `wireframe_optimization_jax.py:858 get_gsco_iteration_jax` | match |
| B-normal assembly | (none) | `wireframe_optimization.py:334 bnorm_obj_matrices` | `wireframe_optimization_jax.py:632 bnorm_obj_matrices_jax` | match (delegated to `WireframeFieldJAX`) |

### Result data structures

| Field | C++ tuple element | JAX dataclass field | Notes |
|---|---|---|---|
| `x` | `Array` (segs × 1) | `WireframeGSCOResult.x` (segs × 1) | C++ returns truncated; JAX returns full-size, sliced by caller via `history_length`. |
| `loop_count` | `IntArray(nLoops)` | `WireframeGSCOResult.loop_count` | match |
| `iter_history` | `IntArray(hist_ind+1)` (sliced) | fixed `(n_iter+1,)`, valid through `history_length` | JIT requires fixed shapes; documented in docstring (line 317-319) |
| `curr_history` | `Array(hist_ind+1)` | same | match |
| `loop_history` | `IntArray(hist_ind+1)` | same | match |
| `f_B/f_S/f_history` | `Array(hist_ind+1)` | same | match |

---

## (a) RCLS Parity

### Algorithm under audit

Both routines solve:

```
min_x  0.5 * (||A x - b||^2 + ||W x||^2)   subject to  C x = d
```

via a null-space parameterization built from the QR factorization of `C^T`:

1. `C^T = Q R` (full QR; `Q ∈ R^{n×n}`, `R ∈ R^{n×p}`).
2. Split `Q = [Q1 | Q2]` with `Q1 ∈ R^{n×p}` (constrained subspace), `Q2 ∈ R^{n×(n-p)}` (null space).
3. Solve `R^T u = d` for `u`.
4. Solve the reduced normal equations `(Q2^T A^T A Q2 + Q2^T W^T W Q2) v = Q2^T A^T b - Q2^T A^T A Q1 u - Q2^T W^T W Q1 u`.
5. `x = Q [u; v]`.

This is mathematically the standard QR-based equality-constrained least-squares with Tikhonov regularization.

### MATH parity

The reduced normal equations in both implementations are formed identically:

CPU (`wireframe_optimization.py:803-814`):
```python
AQ2mat = Amat @ Q2mat
WQ2mat = Wmat @ Q2mat
LHS = AQ2mat.T @ AQ2mat + WQ2mat.T @ WQ2mat

AQ1mat = Amat @ Q1mat
WQ1mat = Wmat @ Q1mat
AQ1uvec = AQ1mat @ uvec
WQ1uvec = WQ1mat @ uvec
AQ2bvec = AQ2mat.T @ bvec
RHS = AQ2bvec - AQ2mat.T @ AQ1uvec - WQ2mat.T @ WQ1uvec
```

JAX (`wireframe_optimization_jax.py:138-147`):
```python
AQ2mat = Amat @ Q2mat
WQ2mat = Wmat @ Q2mat
LHS = AQ2mat.T @ AQ2mat + WQ2mat.T @ WQ2mat

AQ1mat = Amat @ Q1mat
WQ1mat = Wmat @ Q1mat
AQ1uvec = AQ1mat @ uvec
WQ1uvec = WQ1mat @ uvec
AQ2bvec = AQ2mat.T @ bvec
RHS = AQ2bvec - AQ2mat.T @ AQ1uvec - WQ2mat.T @ WQ1uvec
```

**Verdict: math identical.**

### COMPUTATION parity — solver primitives

**Triangular solve for `u`:**
- CPU at `wireframe_optimization.py:801`: `uvec = scipy.linalg.solve_triangular(Rmat.T, dvec, lower=True)` — exploits triangular structure.
- JAX at `wireframe_optimization_jax.py:136`: `uvec = jnp.linalg.solve(Rmat.T, dvec)` — dense LU factorization, no triangular hint.

**Reduced normal-equation solve for `v`:**
- CPU at `wireframe_optimization.py:818`: `vvec = scipy.linalg.lstsq(LHS, RHS)[0]` — by default invokes LAPACK `gelsd` (SVD-based, robust to rank deficiency, default `cond` is machine eps).
- JAX at `wireframe_optimization_jax.py:149`: `vvec = jnp.linalg.lstsq(LHS, RHS, rcond=None)[0]` — `rcond=None` triggers a deprecation/silent default that resolves to machine eps in modern JAX. The underlying implementation lowers to `xla.svd` plus pseudoinverse multiplication; semantics are close to NumPy's `lstsq`, not SciPy's `gelsd`.

**QR factorization:**
- CPU at `wireframe_optimization.py:824 _qr_factorization_wrapper`: wraps `scipy.linalg.qr(M)` and **retries once** on detected NaNs in `R`, per a known scipy/openblas issue (issue 5586 / numpy 20356). Falls back to `RuntimeError` on second NaN.
- JAX at `wireframe_optimization_jax.py:130`: `Qfull, Rtall = jnp.linalg.qr(Ctra, mode="complete")` — no NaN-retry wrapper. The XLA QR path uses Householder reflections and does not exhibit the cited bug, but **the retry semantic is not portable**.

**Severity: MEDIUM (for NaN-retry omission) and INFO (for solver-primitive differences).**

Tests `test_regularized_constrained_least_squares_jax_matches_cpu` (line 236) and `test_regularized_constrained_least_squares_handles_no_constraints` (line 272) pass at `rtol=1e-10, atol=1e-10`. They do not cover near-singular `LHS` (under-determined free-segment systems) where solver-default cutoffs would dominate. **Recommended action:** add a fixture with `len(free_segs) > n_grid` and a deliberately rank-deficient `Amat`, and confirm parity at relaxed `rtol`.

### PHYSICS parity — constraint enforcement

The constraints `C x = d` are **hard** in both paths (null-space projection, not a penalty). Kirchhoff current-conservation at each node is enforced through the constraint matrix produced by `ToroidalWireframe.constraint_matrices(remove_constrained_segments=True)`. Both backends route through the **same wireframe-host code path** to obtain `(C, d)`; the JAX port does not re-derive constraints. **Verdict: identical.**

### Input/shape validation parity

CPU at `wireframe_optimization.py:769-791` raises:
- `"Number of elements in b must match rows in A"`
- `"A and C must have the same number of columns"`
- `"Number of elements in d must match rows in C"`
- `"Number of elements in vector-form W must match columns in A"`
- `"Number of rows and columns in matrix-form W must both equal number of columns in A"`
- `"W must be a scalar, 1d array, or 2d array"`

JAX at `wireframe_optimization_jax.py:121-127`, `82-99` raises the same set with identical messages. **Verdict: match.**

### Wireframe-current mutation

- CPU `rcls_wireframe` (line 537-538) **mutates** `wframe.currents` in place to write the solution.
- JAX `rcls_wireframe_jax` (line 574) **does not mutate** `wframe.currents`. It returns an immutable dataclass.
- The orchestrator `optimize_wireframe_jax` (line 783-784) calls `_write_wireframe_currents(wframe, x)` to perform the mutation downstream.

This is an intentional purity refactor consistent with the JAX worktree's functional-programming guardrail. The end-state of `wframe.currents` after calling `optimize_wireframe_jax` matches the CPU behavior (verified by `test_optimize_wireframe_jax_rcls_matches_public_cpu_and_mutates` at line 363).

---

## (b) GSCO Parity

### Algorithm under audit

GSCO is a single-loop greedy descent over a fixed list of `2*nLoops` candidate moves (the `nLoops` cells in each of two sign directions). At each iteration:

1. **Eligibility filter** — produce a `2*nLoops` vector of signed `loop_current` values for each candidate, zero where ineligible (`free_loops`, `max_loop_count`, `no_new_coils`, `match_current`, `max_current`, `no_crossing`).
2. **Objective sweep** — for each eligible candidate, compute the post-move `2*f_B` (squared residual) and `2*f_S` (active-segment count delta), and the combined `2*f = 2*f_B + λ_S * 2*f_S`.
3. **Argmin** — pick the candidate with smallest `2*f`.
4. **Stopping checks** — `none_eligible`, `undone_loop` (the chosen candidate is the inverse of the previous accepted candidate and the cost would worsen), `last_iter`.
5. **Update** — apply the loop delta to `x`, update `residual = A·x − b`, update `loop_count`, push to history.

The algorithm contract — `f = 0.5 * ||A·x − b||^2 + 0.5 * λ_S * Σ 1{|x_i|>tol}` — is a **discrete L0-penalized least-squares** objective optimized by single-loop swaps. It is **not gradient-based**; correctness depends on the eligibility filter being identical and on the argmin/tie-breaking being deterministic.

### MATH parity (cost, gradient surrogate, greedy update rule)

- **`f_B` definition.** C++ at `wireframe_optimization.cpp:233-235`:
  ```cpp
  two_f_B_pos += (Ax_minus_b_ptr[m] + bnorm) * (Ax_minus_b_ptr[m] + bnorm);
  ```
  JAX at `wireframe_optimization_jax.py:279-280`:
  ```python
  field_delta = jnp.sum(A[:, loop_inds] * loop_delta[None, :, :], axis=2)
  two_f_b = jnp.sum((residual[:, None] + field_delta) ** 2, axis=0)
  ```
  Mathematically identical: `two_f_B = Σ_m (residual_m + Σ_k sign_k · A[m, ind_k] · I)^2`.

- **`f_S` definition.** C++ at `wireframe_optimization.cpp:243-250` and `wireframe_optimization.cpp:371-384`: counts segments where `|x_i| > tol`, accumulates `0.5` per active segment.
  JAX at `wireframe_optimization_jax.py:174-179`: `_gsco_active_entries(x, tol) = where(|x|>tol, 1.0, 0.0)`, `_gsco_two_f_s = sum(...)`. The half-factor and final `0.5*` matches.

- **Total `f`.** Both: `f_total = 0.5 * f_B + 0.5 * λ_S * f_S` — note C++ stores `two_f` and divides by 2 only when recording history (`wireframe_optimization.cpp:308-309`, `319-320`); JAX does the same (`wireframe_optimization_jax.py:430-432`).

- **Greedy update rule.** Both apply `x[ind_tor1] += current; x[ind_pol2] += current; x[ind_tor3] -= current; x[ind_pol4] -= current` (C++ at `wireframe_optimization.cpp:289-292`, JAX at `wireframe_optimization_jax.py:401, 412` via `loop_signs = [1, 1, -1, -1]` and `x.at[loop_inds].add(delta_x)`). The sign convention is identical.

- **Argmin.** C++ uses `std::min_element` (line 259-260), which returns the **first** index in case of ties. JAX uses `jnp.argmin` (line 396), which also returns the first index in row-major order. **Verdict: tie-breaking matches.**

- **`loop_count` increment.** C++ at `wireframe_optimization.cpp:284`: `loop_count(loop_ind) += int(sign)`. JAX at `wireframe_optimization_jax.py:413`: `loop_count.at[loop_ind].add(direction_count)` where `direction_count = where(opt_ind < n_loops, 1, -1)`. Match.

- **Tolerance `tol`.** C++ at `wireframe_optimization.cpp:159`: `tol = 0.001 * default_current`. JAX at `wireframe_optimization_jax.py:336`: `tol = 0.001 * default_current_abs`. The JAX path additionally applies `jnp.abs(default_current)` (line 332). The CPU public wrapper applies `np.abs(default_current)` before the call (`wireframe_optimization.py:661`), so the semantics agree at the public boundary.

**Verdict: MATH parity is preserved across all five points.**

### PHYSICS parity (Kirchhoff conservation under GSCO)

In GSCO, current conservation at each node is **not enforced by a constraint matrix**. Instead, the algorithm only adds **closed loops** of current (4 segments per cell, oriented `+1, +1, -1, -1` around the loop). A single loop addition is locally divergence-free by construction — at each of the four corner nodes, exactly two segments are incremented and two are decremented in the same loop. So Kirchhoff is satisfied **automatically and by superposition**.

C++ encodes this through `loops` (cell-key matrix of segment indices) and `loop_sgns = {1, 1, -1, -1}` (lines 242, 424).
JAX encodes this through `loops_arr` (loaded from `wframe.get_cell_key()`) and the same `loop_signs = jnp.asarray([1.0, 1.0, -1.0, -1.0])` constant (lines 205, 276, 337).

**Verdict: identical.** Kirchhoff is preserved by both backends provided the host `wframe.get_cell_key()` returns a properly-oriented cell key — which it does (legacy CPU coverage in `test_wf_optimization.py`).

### ALGORITHM parity — eligibility filter (`check_eligibility` vs `_gsco_candidate_currents`)

C++ `check_eligibility` (`wireframe_optimization.cpp:393-536`) walks the `2*nLoops` candidates serially (parallelized via `#pragma omp parallel for`). Each candidate is evaluated for:

1. `free_loops` flag (line 403).
2. `max_loop_count`: reject if `|loop_count[i%nLoops] + sign| > max_loop_count` (line 412).
3. `no_new_coils`: reject if no segment in the loop currently carries current (lines 427-439).
4. `match_current`: reject if loop touches segments with inconsistent absolute currents; otherwise adopt the matched absolute current (lines 442-468).
5. `max_current`: reject if any post-move `|x_i + curr_to_add|` exceeds `max_current` (lines 474-485).
6. `no_crossing`: for each toroidal endpoint node, count current-carrying connected segments (with this loop's prospective contribution added in); reject if any node would have > 2 active segments (lines 488-530).

JAX `_gsco_candidate_currents` (`wireframe_optimization_jax.py:182-260`) implements all six checks via vectorized boolean masks:

1. Free-loop flag: line 208.
2. `max_loop_count`: lines 209-211.
3. `no_new_coils`: lines 213-214.
4. `match_current`: lines 216-236 — uses `jnp.max(where(nonzero, abs, 0))` to compute `matched_abs_current` and `any(where(nonzero, abs != matched, False))` for mismatch.
5. `max_current`: lines 240-241.
6. `no_crossing`: lines 243-258 — uses `jnp.argmax(matches, axis=3)` for the "is segment k part of this loop" lookup, then `take_along_axis` to pull the contribution.

**Mathematical equivalence of `match_current`:**

C++ (lines 446-465) iterates the four loop segments; tracks `abs_curr = abs(x[loop_inds[j]])` on the first nonzero entry, sets `mismatch = true` on any subsequent nonzero with different magnitude. If non-mismatch, adopts `loop_curr = sign * abs_curr` (if any nonzero) else `sign * default_current`.

JAX (lines 218-235) computes:
```python
abs_loop_x = jnp.abs(loop_x)                       # (2*nLoops, 4)
nonzero_currents = abs_loop_x > 0.0
matched_abs_current = jnp.max(
    jnp.where(nonzero_currents, abs_loop_x, 0.0),
    axis=1,
)
mismatch = jnp.any(
    jnp.where(nonzero_currents,
              abs_loop_x != matched_abs_current[:, None],
              False),
    axis=1,
)
loop_current = jnp.where(matched_abs_current != 0.0,
                         directions * matched_abs_current,
                         directions * default_current)
eligible = eligible & ~mismatch
```

The two are equivalent: `matched_abs_current` equals the max nonzero (since the C++ "first nonzero" only differs from "max" when there is a mismatch, in which case both fail). The mismatch test compares each nonzero entry against the max — if any disagrees, mismatch fires. **Equivalent.**

**Mathematical equivalence of `no_crossing`:**

C++ iterates the 4 toroidal-endpoint nodes (lines 493-525). For each node, it scans the 4 connected segments. For each connected segment, it checks whether it appears in the candidate loop's 4 indices and accumulates the prospective loop-current contribution (`curr_to_add`). Then it counts how many `|x[seg_k] + curr_to_add| > tol`. Reject if any node yields count > 2.

JAX (lines 244-258):
```python
toroidal_segment_inds = loop_inds[:, [0, 2]]                       # (2*nLoops, 2)
nodes = jnp.reshape(segments[toroidal_segment_inds, :], (2*n_loops, 4))
connected = connections[nodes, :]                                  # (2*nLoops, 4, 4)
loop_deltas = loop_signs[None, :] * loop_current[:, None]          # (2*nLoops, 4)
matches = connected[:, :, :, None] == loop_inds[:, None, None, :]  # (2*nLoops, 4, 4, 4)
first_match = jnp.argmax(matches, axis=3)
matched_delta = jnp.take_along_axis(
    jnp.broadcast_to(loop_deltas[:, None, None, :], matches.shape),
    first_match[:, :, :, None], axis=3)[:, :, :, 0]
current_to_add = jnp.where(jnp.any(matches, axis=3), matched_delta, 0.0)
active_connections = jnp.abs(x[connected] + current_to_add) > tol
crossing_found = jnp.any(jnp.sum(active_connections, axis=2) > 2, axis=1)
eligible = eligible & ~crossing_found
```

This is a faithful vectorization: `connected[i, j, :]` are the 4 segments touching node `j` of candidate `i`; `matches` indicates whether each connected segment is one of the 4 loop segments; `current_to_add` adds the loop's contribution to those that are. The `jnp.argmax(matches, axis=3)` returns 0 when there is no match (since matches is all-False), but `current_to_add` is then zeroed by `jnp.where(any(matches), matched_delta, 0)`. **Equivalent.**

### ALGORITHM parity — stopping conditions

| Condition | C++ | JAX |
|---|---|---|
| `none_eligible` | `wireframe_optimization.cpp:266` `nEligible < 1` (counted from `eligible_curr_ptr[jj] == 0.0`) | `wireframe_optimization_jax.py:404-405` `n_eligible = sum(candidate_currents != 0.0); stop_none_eligible = n_eligible < 1` |
| `undone_loop` | `wireframe_optimization.cpp:270` `i > 0 && (opt_ind + nLoops % (twoNLoops)) == opt_ind_prev` | `wireframe_optimization_jax.py:406` `(iteration > 0) & (opt_ind + n_loops == opt_ind_prev)` |
| `last_iter` | `wireframe_optimization.cpp:279` `i + 1 == nIter` | `wireframe_optimization_jax.py:407` `iteration + 1 == n_iter` |
| Reject undone if worse | `wireframe_optimization.cpp:272-273` `if (two_fs[opt_ind] > two_f_latest) accept_current_loop = false` | `wireframe_optimization_jax.py:408-409` `reject_undo = stop_undone_loop & (two_fs[opt_ind] > two_f_latest)` |

**[HIGH] Note on the `undone_loop` precedence quirk:**

C++ at `wireframe_optimization.cpp:270`:
```cpp
else if (i > 0 && (opt_ind + nLoops % (twoNLoops)) == opt_ind_prev) {
```
Because `%` binds tighter than `+`, this parses as `(opt_ind + (nLoops % twoNLoops)) == opt_ind_prev`. Since `nLoops < twoNLoops`, `nLoops % twoNLoops == nLoops`, so the check is really `(opt_ind + nLoops) == opt_ind_prev`. This correctly catches the case where the previous accepted candidate was `opt_ind` (in [0, nLoops)) and the current pick is the **negative-direction twin** `opt_ind + nLoops`. **But it fails to catch the reverse:** if the previous pick was in the negative direction (`opt_ind_prev ∈ [nLoops, 2*nLoops)`) and the current pick is the positive twin, then `opt_ind + nLoops > opt_ind_prev` always, so the check never fires.

JAX at `wireframe_optimization_jax.py:406` mirrors this exactly: `(opt_ind + n_loops == opt_ind_prev)`. **By design, parity is preserved**, and the parity tests (e.g. `test_gsco_jax_matches_cpp_fixed_state_baseline` line 397) confirm this.

The author would likely want `((opt_ind + n_loops) % (2*n_loops)) == opt_ind_prev` to catch both directions, but porting that without breaking C++ parity would be a downstream upstream bug fix, not a port task. **Recommended action (separate upstream issue):** file a parity-neutral note that both kernels share this latent asymmetry.

### ALGORITHM parity — once-set-stays-set flag latency in C++

C++ initializes `accept_current_loop = true; stop_none_eligible = false; stop_undone_loop = false; stop_last_iter = false` **outside** the for-loop (lines 178-183) and **never resets them at iteration start**. Once `accept_current_loop = false` is set in some iteration, it stays `false` for every subsequent iteration. Likewise the `stop_*` flags.

In practice, this latent bug does not affect output because:
- If `stop_none_eligible` or `stop_undone_loop` (with rejection) fires, `accept_current_loop = false` and the `if(stop_now) break;` at line 325 exits in the same iteration.
- If `stop_undone_loop` fires but the move improves `two_f`, `accept_current_loop` is **not flipped to false** by line 273 (the inner `if` guard skips the assignment), so it retains its prior value — which under "first iteration after acceptance" is still `true`. The accepted loop is then recorded and the iteration breaks via `stop_now`.

The JAX kernel uses fresh per-iteration logic in `lax.scan` (lines 405-410) with `accept_loop = (~done) & (~stop_none_eligible) & (~reject_undo)`, plus a sticky `done` flag (line 446 `done_next = done | stop_now`). This is semantically equivalent and arguably cleaner, but **the parity tests do not probe this corner-case**: a scenario where C++'s sticky flag latency would diverge from the JAX scan would require a contrived `stop_undone_loop`-then-recover sequence that the current fixture does not exercise.

**Severity: LOW** — purely a code-hygiene concern in C++; behavior identical on tested inputs.

### COMPUTATION parity — dtype and reduction order

- **dtype.** C++ uses `double` throughout. JAX uses `_as_jax_float64` for `A`, `b`, `x_init` (lines 321-323) and `_as_jax_int32` for the integer arrays (lines 324-328). The integer width differs (C++ `int`, JAX `int32`). For the wireframe sizes in use (`n_loops < 2^31`), this is irrelevant. **OK.**

- **`A·x − b` running residual.** C++ accumulates `Ax_minus_b` per grid point via a serial inner loop (lines 117-122). The JAX kernel does this in batch: `residual0 = A @ x0 - b` (line 339). Reduction order over the segment-axis differs (XLA matmul reduction order is implementation-defined). This is the primary source of any potential `ulp`-scale drift in `f_B`.

- **Candidate `two_f_b` reduction.** C++ accumulates `two_f_B_pos` as a serial scalar sum over `m ∈ [0, nGrid)` (lines 225-235). JAX reduces over `axis=0` (grid axis) of a `(nGrid, 2*n_loops)` matrix (line 280). On CPU XLA this is typically a pairwise tree reduction; on CUDA it would be a grid-strided reduction. In the worst case this is `O(log nGrid) ulp` divergence per candidate.

- **`A[:, loop_inds]` access.** JAX builds a `(nGrid, 2*n_loops, 4)` slice each iteration. For tiny `n_loops` (test fixture: 2), this is harmless. For large wireframes (`O(10^3)` loops × `O(10^3)` grid points), the gather pattern is a memory cost roughly equal to the matrix-vector product itself.

- **Argmin determinism.** Both back-ends return the **first** index in case of ties (`std::min_element` and `jnp.argmin` agree on lowest-index tie-break under row-major XLA layout). **OK.**

**Severity: MEDIUM** for reduction-order drift on `f_B`; tests pass at `rtol=1e-10` for the seeded fixtures used (very small `nGrid=5`, `nLoops=2`). **Recommended action:** add a larger fixture (`nGrid=200`, `nLoops=50`) and assert at a relaxed `rtol=1e-9` to characterize drift on more realistic problems; alternatively rely on the `parity_ladder` derivative-heavy lane tolerances for greedy paths.

### Output-shape contract divergence (documented)

C++ returns truncated history arrays via `xt::view(..., xt::range(0, hist_ind+1))` (lines 340-348). JAX returns fixed-size `(n_iter+1,)` arrays plus a `history_length` scalar; the wrapper `optimize_wireframe_jax` slices them on host before returning (lines 815-836). The Item 31 test `_compare_gsco_result` (line 139) does the slicing explicitly. **Documented at `wireframe_optimization_jax.py:317-319`.** Semantic equivalence confirmed by the parity tests.

---

## (c) Constraint Parity

### RCLS — null-space projection (hard constraints)

Both `regularized_constrained_least_squares` and `regularized_constrained_least_squares_jax` enforce `C x = d` exactly through the null-space parameterization. No penalty, no Lagrange multiplier seam. The constraint matrices `(C, d)` come from the host `ToroidalWireframe.constraint_matrices(remove_constrained_segments=True)` — **the same host code path for both backends**. There is no JAX-port of the wireframe's constraint-matrix assembly; the JAX RCLS path is a numerical wrapper around a host-supplied `(C, d, free_segs)` triple.

The wireframe's underlying constraints encode Kirchhoff (`Σ I = 0` at each node), Amperian-loop conditions (poloidal/toroidal current), and explicit per-segment zeroing — all built on the CPU side in `ToroidalWireframe`. **Coverage: identical because both backends share the upstream assembly.**

### GSCO — implicit constraint preservation

GSCO does not maintain a constraint matrix at all. As discussed in section (b)/PHYSICS, the algorithm preserves Kirchhoff conservation automatically by only adding closed loops, and it preserves segment-current constraints (e.g., `set_segments_constrained([...])` in the legacy CPU test) by populating `free_loops` such that ineligible cells are pre-filtered.

**Constraint coverage in tests:**
- `test_wf_optimization.py::test_toroidal_wireframe_rcls` (lines 57-228) exercises poloidal-current, toroidal-current, and segment-zero constraints, plus the `check_constraints()` assertion (line 102, 112, 131). This test runs on the CPU path only.
- `test_wireframe_optimization_jax_item31.py::test_rcls_wireframe_jax_matches_cpu_without_mutating_wireframe` (line 299) uses a stub `_WireframeFixture` with a hand-crafted `C, d` pair. It does **not** exercise the upstream constraint assembly.
- `test_optimize_wireframe_jax_rcls_matches_public_cpu_and_mutates` (line 363) uses the full `ToroidalWireframe` but with `assume_no_crossings=False` and no special constraints set. The default `constraint_matrices(assume_no_crossings=False)` includes Kirchhoff for all nodes.

**Gap: there is no JAX-vs-CPU parity test that explicitly constructs a wireframe with `set_poloidal_current`, `set_toroidal_current`, or `set_segments_constrained`, runs RCLS through both backends, and compares.** The math IS identical (because both backends use the same host-supplied `(C, d)`), but a regression-shielding test would catch any future refactor of the orchestrator that, e.g., bypasses `constraint_matrices`. **Severity: INFO.**

### Constraint-matrix dimension checks

CPU `rcls_wireframe` at line 502: `if np.shape(C)[0] >= len(free_segs): raise ValueError(...)`.
JAX `rcls_wireframe_jax` at line 598: identical check, identical message.

CPU `regularized_constrained_least_squares` at lines 769-775: shape checks for `(A, b, C, d)`. JAX mirror at lines 121-127. **Match.**

---

## (d) Convergence-Criteria Parity

### RCLS

RCLS is a **direct solver** — there is no iterative convergence criterion. The only "convergence" is the LAPACK lstsq cutoff used to handle rank-deficient `LHS`. As noted in section (a), CPU uses `scipy.linalg.lstsq` defaults (gelsd, cond≈eps) and JAX uses `jnp.linalg.lstsq(rcond=None)` (gelss-like, cond≈eps). For well-conditioned `LHS`, both return the same minimum-norm solution. For rank-deficient `LHS`, the two paths may diverge by `O(ulp · κ(LHS))` in the chosen basis.

**Severity: INFO** — no tests probe near-singular regime.

### GSCO

GSCO terminates on **three deterministic conditions**:
1. `none_eligible` — no candidate has nonzero `loop_current`.
2. `undone_loop` — the chosen candidate is the inverse of the previous accepted candidate (subject to the precedence quirk noted above) AND the move would worsen `two_f`.
3. `last_iter` — reached `nIter` iterations.

There is no convergence tolerance. The history is exact (modulo `f_B` reduction-order drift). **Parity is preserved on all three conditions** as tabulated in section (b)/Stopping-Conditions.

A subtler point: **the C++ and JAX agree on what counts as an iteration**. JAX `lax.scan` runs exactly `n_iter` scan steps even after `done=True`, but the carry is fixed once `done`. The `history_length` returned to the host is the same as C++'s `hist_ind + 1`. **OK.**

### Tolerance for `f_S` quantization

Both backends use `tol = 0.001 * abs(default_current)` for the `|x_i| > tol` predicate inside `f_S`. This means **a floating-point `x_i` near `tol`** is binary-classified, and a `ulp`-scale drift in `x_i` between the two backends could flip the count and cause `f_S` to differ by 1.0 (since each segment contributes 1.0 to `2*f_S`).

In the parity tests (`_gsco_problem` line 94), `default_current = 0.2` and `x_init = 0`, so `x` stays a small multiple of 0.2 throughout. The tolerance is `2e-4`, well above any `ulp` drift in `0.2` (≈4e-17). **Parity holds on the test fixtures.** A pathological fixture (e.g. `default_current=1e-12`) could plausibly trigger a count flip; this is also outside the scope of typical use.

**Severity: INFO** — document the threshold sensitivity in the JAX module docstring; no fix recommended.

---

## Test Coverage Gaps

| Concern | Coverage | Gap |
|---|---|---|
| RCLS, scalar W | `test_regularized_constrained_least_squares_jax_matches_cpu[W0]` | OK |
| RCLS, vector W | `test_regularized_constrained_least_squares_jax_matches_cpu[W1..W3]` | OK |
| RCLS, full matrix W | `test_regularized_constrained_least_squares_jax_matches_cpu[W4]` | OK |
| RCLS, no constraints (`C.shape[0] == 0`) | `test_regularized_constrained_least_squares_handles_no_constraints` | OK |
| RCLS, rank-deficient `LHS` | **none** | **INFO** — solver-default cutoff divergence untested |
| RCLS, NaN-retry semantic (`_qr_factorization_wrapper`) | **none** in JAX path | **MEDIUM** — JAX has no NaN retry; if XLA QR ever returns NaN, the JAX path will not recover |
| RCLS, set_poloidal_current / set_toroidal_current end-to-end | **none in JAX parity test** (legacy CPU test only) | **INFO** — orchestrator regression risk |
| RCLS, public `optimize_wireframe_jax` mutation | `test_optimize_wireframe_jax_rcls_matches_public_cpu_and_mutates` | OK |
| RCLS, immutability of input `wframe` for the pure helper | `test_rcls_wireframe_jax_matches_cpu_without_mutating_wireframe` (line 338 asserts `fixture.currents` unchanged) | OK |
| GSCO, baseline fixed-state | `test_gsco_jax_matches_cpp_fixed_state_baseline` | OK |
| GSCO, all eligibility flag combinations | `test_gsco_jax_matches_cpp_eligibility_options[no_crossing/no_new_coils/match_current/max_loop_count]` (3 parametrized cases) | OK for the three combinations; not exhaustive (`max_current` boundary not parametrized independently) |
| GSCO, `gsco_wireframe_jax` wrapper + `_GSCOFixture` | `test_gsco_wireframe_jax_wrapper_matches_cpp_without_mutating_wireframe` | OK |
| GSCO, public `optimize_wireframe_jax` mutation + iteration helper | `test_optimize_wireframe_jax_gsco_matches_public_cpu_and_iteration_helper` | OK |
| GSCO, `x_init=None, loop_count_init=None` shape branch | `test_optimize_wireframe_jax_gsco_accepts_present_none_initial_state` | OK |
| GSCO, jit + transfer-guard | `test_gsco_jax_jits_under_transfer_guard` | OK |
| GSCO, `stop_undone_loop` actually triggered | **none** (the fixtures terminate by `last_iter` or `none_eligible`) | **INFO** — the undo-detection branch is not exercised in any test |
| GSCO, `stop_none_eligible` actually triggered | **none** explicit | **INFO** |
| GSCO, large fixture (`nGrid ≥ 100`, `nLoops ≥ 50`) at relaxed rtol | **none** | **MEDIUM** — reduction-order drift uncharacterized at scale |
| GSCO, `default_current ≈ tol` edge case | **none** | **INFO** — `f_S` count-flip risk uncharacterized |
| `get_gsco_iteration_jax` parity | `test_optimize_wireframe_jax_gsco_matches_public_cpu_and_iteration_helper` (one iteration) | OK |
| `bnorm_obj_matrices_jax` | `test_bnorm_obj_matrices_jax_matches_cpu_public_surface_mode[area_weighted=True/False]` | OK |
| `bnorm_obj_matrices_jax` with `ext_field` or `bnorm_target` | **none** | **INFO** — branch coverage missing |

---

## Recommended Actions (ordered by severity)

### HIGH
- **R-1.** File an upstream bug (not a port fix) noting the C++ operator-precedence asymmetry in the `stop_undone_loop` test at `wireframe_optimization.cpp:270`. The JAX port mirrors the bug to preserve parity. Recommended fix in **both** would be `((opt_ind + n_loops) % (2*n_loops)) == opt_ind_prev`. Coordinate with upstream `simsopt` before fixing only on the JAX side.

### MEDIUM
- **R-2.** Add a JAX-side NaN-retry guard around `jnp.linalg.qr` to mirror `_qr_factorization_wrapper`'s defensive retry, or assert in the JAX module docstring that XLA QR is reliable and the wrapper is intentionally not ported.
- **R-3.** Add a large-fixture GSCO parity test (e.g. `nGrid=128`, `nLoops=64`, 25 iterations) at `rtol=1e-9` to characterize reduction-order drift in `f_B_history` and document the bound.
- **R-4.** Add an RCLS rank-deficient-LHS test (e.g. duplicate columns in `A`, no regularization, no constraints) to characterize solver-cutoff divergence between SciPy `gelsd` and JAX `jnp.linalg.lstsq`.

### LOW
- **R-5.** Refactor the C++ stopping-flag declarations to live inside the for-loop so the latent "sticky flag" issue cannot bite a future maintainer (cosmetic; not a parity fix). This is upstream-territory; the JAX scan already does this correctly.
- **R-6.** Document the `default_current ≈ tol` edge case in the JAX module docstring (no code fix needed).

### INFO
- **R-7.** Add JAX-side parity tests for `bnorm_obj_matrices_jax` with `ext_field` and `bnorm_target` non-None branches.
- **R-8.** Add a `stop_undone_loop`-triggering fixture (e.g. a tight-loop two-cell wireframe where the greedy will oscillate immediately) to exercise the undo branch end-to-end.
- **R-9.** Add a JAX parity test that constructs a `ToroidalWireframe` with `set_poloidal_current`, `set_toroidal_current`, and `set_segments_constrained` set, runs RCLS through both backends, and asserts `wf.check_constraints()` plus solution parity.

## Net Verdict

The JAX port of GSCO is **faithful and parity-preserving**, including a deliberate mirror of an upstream operator-precedence quirk in the `stop_undone_loop` check. The cost function, eligibility filter (all six branches), greedy update, history shape contract, and stopping conditions are mathematically and operationally equivalent to the C++ kernel.

The JAX port of RCLS is **mathematically identical** to the SciPy reference, with two solver-primitive substitutions (`scipy.linalg.solve_triangular` → `jnp.linalg.solve`, `scipy.linalg.lstsq` → `jnp.linalg.lstsq`) that produce identical results on well-conditioned fixtures but may diverge by `O(ulp · κ)` on rank-deficient `LHS`. The NaN-retry safeguard in `_qr_factorization_wrapper` is not ported.

No CRITICAL or HIGH-severity parity bug was identified in either port. All issues are MEDIUM or below, and most are matched by INFO-level coverage gaps that would shield against future regressions.
