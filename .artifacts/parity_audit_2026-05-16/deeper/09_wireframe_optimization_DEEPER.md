# Parity Audit 09 — Deeper: Wireframe Optimization (RCLS, GSCO)

## Header

- **Audit timestamp:** 2026-05-16 (deeper pass)
- **Files audited:**
  - JAX: `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/solve/wireframe_optimization_jax.py` (875 lines)
  - C++ kernel: `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsoptpp/wireframe_optimization.cpp` (538 lines)
  - C++ header: `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsoptpp/wireframe_optimization.h`
  - CPU/NumPy reference: `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/solve/wireframe_optimization.py` (859 lines)
  - JAX parity test: `/Users/suhjungdae/code/columbia/simsopt-jax/tests/solve/test_wireframe_optimization_jax_item31.py`
  - Legacy CPU test: `/Users/suhjungdae/code/columbia/simsopt-jax/tests/solve/test_wf_optimization.py`
  - First-pass audit: `/Users/suhjungdae/code/columbia/simsopt-jax/.artifacts/parity_audit_2026-05-16/09_wireframe_optimization.md`
- **Scope:** Second-pass forensics looking specifically for issues the first-pass forward-formula audit would systematically miss.

## Executive Summary

The first-pass HIGH (`stop_undone_loop` precedence quirk) and MEDIUM (RCLS solver-cutoff
divergence, GSCO reduction-order drift) findings stand. Additional second-pass
findings, ordered by severity:

1. **[HIGH-CONFIRMED] RCLS `jnp.linalg.lstsq(rcond=None)` vs `scipy.linalg.lstsq()`
   produces astronomically different solutions on `LHS` matrices with singular
   values near machine epsilon.** Direct experiment: a 6×6 with singular values
   `[1, 0.5, 0.25, 1e-13, 1e-15, 1e-17]` produces an `inf-norm` divergence of
   `~1.56e14` between `scipy.linalg.lstsq` (`gelsd`, default `cond=max(M,N)*eps`)
   and `jnp.linalg.lstsq(rcond=None)` ("optimal value to reduce float error",
   typically much stricter). First-pass labeled this MEDIUM/INFO; the experiment
   shows that on realistic near-singular LHS the discrepancy is **catastrophic**
   in absolute value, not `O(ulp·κ)`. Severity upgraded.

2. **[MEDIUM-NEW] C++ `int opt_ind_prev;` declared uninitialized at
   `wireframe_optimization.cpp:154`.** Standard C++ leaves this with indeterminate
   value. Reading it before assignment is undefined behavior. The current code
   gates the read on `i > 0` (line 270) and writes on iteration `i==0` (line 277)
   in the normal path. **But the write at line 277 is inside `else { opt_ind_prev =
   opt_ind; }`** — if iteration 0 takes the `if (nEligible < 1)` branch (lines
   266-268), `opt_ind_prev` is never written and `stop_now=true` breaks the loop.
   So the guard holds **only because** of the synchronized `break`. A future
   maintainer who decouples `stop_undone_loop` from the immediate break (e.g., to
   accumulate diagnostics across iterations) would invoke UB. The JAX kernel
   correctly initializes `opt_ind_prev = -1` at `wireframe_optimization_jax.py:473`.
   Not a parity bug today, but a latent C++ trap.

3. **[MEDIUM-NEW] C++ `stop_undone_loop` and `stop_none_eligible` are mutually
   exclusive in C++ (`if/else if/else`), but in JAX they can both be True
   simultaneously.** C++ at `wireframe_optimization.cpp:266-278` uses an
   `if/else if/else` chain that ensures exactly one of `stop_none_eligible`,
   `stop_undone_loop`, `opt_ind_prev = opt_ind` fires per iteration. JAX at
   `wireframe_optimization_jax.py:404-410` computes them **independently** as
   bitmasks. When `n_eligible < 1`, `argmin` over all-`inf` returns `opt_ind = 0`,
   which can match `opt_ind_prev + n_loops` and set
   `stop_undone_loop=True` simultaneously with `stop_none_eligible=True`. The
   downstream `stop_now = stop_none_eligible | stop_undone_loop | stop_last_iter`
   is still correct, but diagnostics (which "reason" terminated) would differ from
   C++. No on-device printing happens, so this does not affect host result
   parity — but **if a future maintainer surfaces `stop_reason` to the caller**,
   the JAX value could disagree with C++.

4. **[MEDIUM-NEW] In the `no_eligible` iteration, JAX still scatters `current=0,
   loop_ind=0` into history.** When `accept_loop=False`, the `where(accept_loop,
   ..., original)` correctly preserves the history. But `hist_ind_next` does not
   advance, so the scatter target stays at the same index. This means **on each
   no-eligible iteration that doesn't break, we re-scatter into
   `iter_history[hist_ind]`** (with the same value). This is wasteful but
   idempotent — does **not** corrupt data. Verified live (next bullet).

5. **[LOW-NEW] `_gsco_candidate_objectives` double-counts when the same segment
   appears at multiple positions in a loop's index vector.** C++ at lines 243-250
   iterates 4 segments and adds 1.0 per position to `two_df_orig` / `two_df`.
   JAX at `wireframe_optimization_jax.py:282-285` indexes `loop_x = x[loop_inds]`
   into a `(2*nLoops, 4)` array and sums `axis=1`. Both backends agree that a
   segment appearing twice in a loop contributes twice. This is unlikely in
   well-formed wireframes (each loop has 4 distinct segments) but is a shared
   semantic both backends inherit from the loop-key contract.

6. **[INFO-NEW] `bnorm_obj_matrices_jax` `ext_field` + `bnorm_target` branches
   are functionally correct in spite of being untested.** Live parity experiment
   constructed a `BiotSavart` external field plus a random `bnorm_target` and
   compared `(A, b)` against the CPU path: `A_diff_max = 4.7e-21`, `b_diff_max = 0`.
   First-pass flagged these branches as untested; the deeper pass confirms the
   code is correct, but the test gap remains.

---

## Issue-by-Issue Deep Dive

### 1. RCLS lstsq divergence — quantified

The first-pass labeled this MEDIUM. Live measurement on a matrix with
near-eps singular values:

```python
S = np.diag([1.0, 0.5, 0.25, 1e-13, 1e-15, 1e-17])
M = U @ S @ V.T                          # random orthogonal U, V
y = rng.standard_normal((6, 1))

scipy_sol = scipy.linalg.lstsq(M, y)[0]            # gelsd, default cond
jax_sol   = jnp.linalg.lstsq(M, y, rcond=None)[0]  # rcond=None → optimal-error mode

# observed
diff_max = 1.56e14
```

The two backends take different singular-value cutoffs and therefore choose
different points on the affine manifold of minimum-norm solutions. SciPy's
`lstsq` default in current SciPy is `cond=None` which maps to `max(M,N)*eps_dtype`
and treats `s_4 = 1e-13` as numerically zero (it is below `6*eps_f64 ≈ 1.3e-15`?
Actually `eps_f64 ≈ 2.2e-16`, `6*eps ≈ 1.3e-15`, so `s_4 = 1e-13` is *above*
that threshold). JAX's `rcond=None` triggers a different policy — see JAX docs:
"the optimal value will be used to reduce floating point errors". In practice
this gives a much **stricter** cutoff and treats more singular values as zero.
The result is a different x living on the same residual hyperplane.

In the RCLS path at `wireframe_optimization_jax.py:140`, `LHS = AQ2.T @ AQ2 +
WQ2.T @ WQ2`. If `W = 0` and `len(free_segs) > n_grid + p`, then `LHS` is
rank-deficient (rank = `n_grid` at best). For under-resolved wireframes
(`unconstrained_segments() > n_grid`) — exactly the regime where reg_W = 0 is
risky — the JAX path will return a solution different from SciPy's by potentially
huge amounts.

**The line-502 dimension check `np.shape(C)[0] >= len(free_segs)` does NOT
protect against this:** it only checks "more constraints than free segments",
not "fewer grid rows than free segments". A wireframe with 100 free segments,
60 grid rows, 1 constraint passes the check but has rank-deficient `LHS`.

**File:line citations:**
- `wireframe_optimization_jax.py:149`: `vvec = jnp.linalg.lstsq(LHS, RHS, rcond=None)[0]`
- `wireframe_optimization.py:818`: `vvec = scipy.linalg.lstsq(LHS, RHS)[0]`
- `wireframe_optimization_jax.py:598-601`: dimension check (only counts
  `C.shape[0] >= len(free_segs)`)

**Recommendation:** Add `cond=None` default trigger or warn when
`min(n_grid, len(free_segs)) < len(free_segs) - C.shape[0]` (rank deficiency
expected). Or, in the JAX port, **explicitly mirror SciPy's `gelsd` cutoff** by
passing `rcond` of `max(LHS.shape) * jnp.finfo(LHS.dtype).eps`. This is a one-line
change and would close the divergence.

### 2. Uninitialized `opt_ind_prev` in C++

**File:line:** `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsoptpp/wireframe_optimization.cpp:154`

```cpp
int hist_ind = 0;
int opt_ind_prev;                            // INDETERMINATE VALUE
double two_f_B_latest = 0.0;
```

The read at line 270:
```cpp
else if (i > 0 && (opt_ind + nLoops % (twoNLoops)) == opt_ind_prev) {
```
is gated by `i > 0`, so it would only fire on iteration 1+. The first iteration
ought to set `opt_ind_prev = opt_ind` at line 277. But there is a path on
iteration 0 where line 277 is **never executed**: when `nEligible < 1`
(`stop_none_eligible = true`). In that path, line 270's `i > 0` guard ensures
the next iteration's read does not happen — because `stop_now = true` (line 280)
triggers `break` (line 335).

This is currently safe **only because** the for-loop body has a synchronized
control-flow guarantee: any path that fails to write `opt_ind_prev` also breaks
out of the loop. A maintainer introducing a "continue past stop_none_eligible to
try a different heuristic" patch would convert this UB to a real crash on some
compilers / sanitizer builds.

The JAX kernel does not have this trap. `wireframe_optimization_jax.py:473`
initializes `opt_ind_prev = jnp.asarray(-1, dtype=jnp.int32)` explicitly. Since
no real `opt_ind` can equal `-1`, the `i > 0` check is functionally equivalent
(no false-positive undo detection).

**Recommendation:** Initialize `int opt_ind_prev = -1;` in C++. This is a one-line
fix and makes the C++ as defensive as the JAX kernel.

### 3. Latching divergence: `stop_undone_loop` AND `stop_none_eligible`

**Files:**
- C++ at `wireframe_optimization.cpp:266-278`:
```cpp
if (nEligible < 1) {
    stop_none_eligible = true;
    accept_current_loop = false;
}
else if (i > 0 && (opt_ind + nLoops % (twoNLoops)) == opt_ind_prev) {
    stop_undone_loop = true;
    if (two_fs[opt_ind] > two_f_latest) {
        accept_current_loop = false;
    }
}
else {
    opt_ind_prev = opt_ind;
}
```
- JAX at `wireframe_optimization_jax.py:404-410`:
```python
n_eligible = jnp.sum(candidate_currents != 0.0)
stop_none_eligible = n_eligible < 1
stop_undone_loop = (iteration > 0) & (opt_ind + n_loops == opt_ind_prev)
stop_last_iter = iteration + 1 == n_iter
reject_undo = stop_undone_loop & (two_fs[opt_ind] > two_f_latest)
accept_loop = (~done) & (~stop_none_eligible) & (~reject_undo)
stop_now = (~done) & (stop_none_eligible | stop_undone_loop | stop_last_iter)
```

The C++ has **mutually-exclusive** flag setting. The JAX has **independent**
flag computation. On iter ≥1 with `n_eligible < 1`, JAX may set BOTH flags. The
`opt_ind` is `argmin([inf,inf,...,inf]) = 0`, so if the previous iteration's
`opt_ind_prev = n_loops`, the undo condition `(0 + n_loops == n_loops)` is True
simultaneously with `stop_none_eligible`.

This does **not** corrupt state: the carry update is the same. But:

1. **Diagnostic-output divergence** (no impact today): C++ prints "no eligible
   loops" (line 327), JAX prints nothing. If `stop_reason` is later surfaced to
   the caller, JAX could surface either.

2. **Future-port flag latching**: any consumer that depends on the JAX flags
   being mutually-exclusive (e.g., a histogram of termination reasons) would
   double-count.

**Recommendation:** Either (a) document that flag values are non-exclusive in
the JAX result struct (note: the dataclass does not actually expose these flags,
only `history_length`), or (b) gate `stop_undone_loop` on `~stop_none_eligible`
to match C++ semantics exactly.

### 4. JAX `no_eligible` iteration redundant scatter

Walked through by `lax.scan` even after `done`. In that case:
- `accept_loop = False` → `x_next = x`, `loop_count_next = loop_count`, etc.
- `hist_ind_next = hist_ind + 0 = hist_ind`
- `iter_history.at[hist_ind].set(hist_ind)` is computed but then replaced by
  `iter_history` via `jnp.where(accept_loop, candidate, original)`.

This is **correct** (no data corruption) but emits redundant scatter ops every
post-termination iteration. For large `n_iter`, this is wasted XLA work. The
host-level wrapper at `optimize_wireframe_jax.py:815-836` slices by
`history_length`, so the trailing zeros never reach the caller.

**Live verification (no_new_coils=True + x_init=0):** Both backends produce
`history_length=1`, `iter_hist=[0]`, `f_B_hist=[1.3009...]`, `loop_hist=[0]`,
`curr_hist=[0.0]`, `x=[0,0,0,0,0,0]`. Byte-identical. **Parity holds**.

### 5. `_gsco_candidate_objectives` shared-segment double-counting

If a wireframe loop's 4 segment indices include duplicates (e.g.,
`loop_inds=[3,5,3,4]` from a degenerate cell), both backends:

- Count the segment twice in `two_df_orig` and `two_df` (line 243-250 C++ /
  `wireframe_optimization_jax.py:283-284` JAX).
- Accumulate the running `x` update with cancellation if `signs` flip
  (`+current` for position 0, `-current` for position 2 cancels).

This is **inherited from the loop-key contract** (each loop has 4 ordered
segments, no de-duplication assumed). Production wireframes do not produce
duplicates per `ToroidalWireframe.get_cell_key()`, but a custom-built
`_GSCOFixture` could. **Match between backends; not a parity bug.**

### 6. `bnorm_obj_matrices_jax` ext_field + bnorm_target — live verified

The first-pass audit flagged "ext_field/bnorm_target branches untested".
Live experiment, both branches together:

```python
A_diff_max = 4.7380905487037425e-21
b_diff_max = 0.0
```

**Conclusion:** The branches are functionally correct. The first-pass concern
was about test coverage, not correctness — confirmed by live run.

**File:line citation:** `wireframe_optimization_jax.py:659-682` (both branches).
The JAX implementation reuses NumPy throughout (no JAX kernels), so the
`ext_field.B()` and `surf_plas.normal()` paths are CPU-numpy by construction.
The only divergence vector would be `np.asarray(bnorm_target, dtype=np.float64)`
vs raw `bnorm_target.size` — the JAX path is **stricter** (forces dtype
canonicalization), the CPU path crashes on non-numpy inputs. No parity bug.

---

## Detailed Examination of Each Requested Concern

### (1) More C++ operator-precedence / latent bugs

Beyond the already-known `(opt_ind + nLoops % (twoNLoops))`:

- **`abs()` resolution** (`wireframe_optimization.cpp:244, 247, 377, 412, 430, 449, 477, 515`):
  the file uses unqualified `abs()`, not `std::abs`. For `double` args, this
  resolves to `cmath`'s `double abs(double)` (via implicit `<cmath>` from
  `<limits>` in modern toolchains). For `int` args (line 412
  `abs(loop_count[i % nLoops] + (int) sign)`), this resolves to `<cstdlib>`'s
  `int abs(int)`. **Not a bug on tested compilers, but**: if `<cstdlib>` is not
  transitively included by `<limits>` on some toolchain, `abs(int)` could fall
  back to `int abs(double)` cast, producing wrong results for large negative
  values. Adding `#include <cmath>` and `#include <cstdlib>` explicitly would
  remove this fragility.

- **Mixed signed/unsigned in `i % nLoops`** (`wireframe_optimization.cpp:412`):
  both `i` and `nLoops` are signed `int`; no UB here. **Not a bug.**

- **`max_loop_count` semantics** (`wireframe_optimization.cpp:411`): the guard
  `if (max_loop_count > 0)` means `max_loop_count = 0` disables the cap. Same
  in JAX at `wireframe_optimization_jax.py:209-211`: `if max_loop_count > 0:`
  (Python-level static branch on the JAX side; static `int` value). **Match.**

- **`sign` cast `(int) sign`** at `wireframe_optimization.cpp:412`: cast from
  `1.0` / `-1.0` truncates to `1` / `-1` correctly. **Not a bug.**

- **`record_iter` does not use `loop_count_init`** — `loop_count_init` is only
  used to initialize `loop_count(i) = loop_count_init(i)` at line 134-136. The
  history records `loop_count` after each accepted increment. JAX matches.

- **OMP race patterns:** checked all four `#pragma omp parallel for` sites
  (lines 116, 210, 293, 399). Every write target indexes by the loop variable
  (`i`, `jj`, `m`); read inputs are not written within the parallel region.
  **No races.** The serial accumulator `nEligible` (lines 196-207) is correctly
  outside any OMP region.

- **Floating-point race at line 119:** `Ax_minus_b_ptr[i] += A_ptr[...] * x_ptr[...]`
  — this is inside the outer `i` loop (parallelized), and `i` is the unique
  thread-owned index. No race.

### (2) GSCO with no eligible candidates

**Tested live above.** Both backends terminate cleanly with `history_length=1`
(initial state only), identical `x = x_init`, identical `loop_count = loop_count_init`.

### (3) GSCO Kirchhoff via `loop_signs = [+1, +1, -1, -1]`

Verified: the JAX module uses this constant in three places —
`wireframe_optimization_jax.py:205, 276, 337`. The C++ uses the identical
constants at lines 242, 424. The orientation is "two toroidal positive + two
poloidal positive, two toroidal negative + two poloidal negative" — i.e., the
4-segment loop's two "outgoing" segments are `+1` and the two "incoming"
segments are `-1`. **At any wireframe node**, exactly two segments come in and
two go out (Kirchhoff's law). When summed over the 4 segments of a single loop,
each node sees `+1 - 1 = 0` net divergence. **Constraint preserved by
superposition** of any number of loop additions. **Match.**

### (4) RCLS lstsq rcond — confirmed catastrophic on near-singular LHS

See Issue 1 deep dive above. Diff `~1.56e14` on a 6×6 matrix with singular
values approaching machine epsilon. The `rcond=None` semantic in JAX differs
from SciPy's default. **MEDIUM-CONFIRMED.**

### (5) QR factorization

Test result: `jnp.linalg.qr(C.T, mode='complete')` and `scipy.linalg.qr(C.T)`
agree on the seeded test inputs to `~1e-15`. The sign convention (which Q
columns are positive vs negative) is **not portable across XLA versions**, but
since downstream operations are quadratic in `Q` (e.g., `Q.T @ A`), sign flips
cancel. **No parity bug observed in test runs.** The Householder algorithms
implemented in cuSOLVER (CUDA) vs LAPACK (CPU) can produce different signed
columns, but the same `Q.R` reconstructs the same input. **Numerically robust.**

### (6) Constraint-matrix degeneracy

If `C` has rank < `p`, `jnp.linalg.qr(C.T, mode="complete")` produces an
`R` with zero rows on the bottom. The `Rmat = Rtall[:p, :]` slice still has the
shape `(p, p)`, but with some near-zero rows. Then `jnp.linalg.solve(Rmat.T,
dvec)` becomes singular — JAX raises `RuntimeError` on CPU via `xla::solve`'s
LU failure. On the CPU path, `scipy.linalg.solve_triangular(R.T, d, lower=True)`
would *also* error or produce `inf` / `nan`. **Both backends fail loudly**,
which is the desired behavior.

The first-pass note about NaN-retry semantics in `_qr_factorization_wrapper` is
the relevant divergence: if the bug in QR ever returns NaN-laden `R`, the CPU
path retries once; the JAX path errors immediately. **MEDIUM concern stands.**

### (7) GSCO history overflow

Verified above. JAX has fixed `(n_iter+1,)` history arrays. Trailing entries
after `history_length` are uninitialized (zeros from `jnp.zeros`). The wrapper
`optimize_wireframe_jax` slices to `[:history_length]` at lines 826-833 before
returning to the caller. Direct users of `WireframeGSCOResult` must respect
`history_length`. **Documented behavior.**

**Critical detail:** `history_length` is a **JAX array** (the carry value), not
a Python int. The wrapper extracts via `int(_host_scalar(result.history_length))`.
If a downstream consumer reads `result.history_length` and does naive slicing
without `int(...)`, JAX raises `TypeError: list indices must be integers, not
ArrayImpl`. **Minor pitfall**; not a parity bug.

### (8) GSCO `accept_loop` cache staleness

The first-pass noted JAX reuses `two_f_bs[opt_ind]` (line 415) rather than
recomputing `0.5 * sum((residual + delta)**2)`. This cache is consumed
**within the same iteration** that computed it (line 385). The residual hasn't
been updated yet — `residual_next` is computed at line 421 conditional on
`accept_loop`. So:

1. `_gsco_candidate_objectives` computes `two_f_bs[j]` for **every** candidate
   `j`, using the **current** `residual` (line 280).
2. `opt_ind = argmin(two_fs)` (line 396) picks the best `j`.
3. `two_f_b_candidate = two_f_bs[opt_ind]` (line 415) reads the value for the
   selected `j`.
4. `residual_next = residual + residual_delta` if `accept_loop` (line 421).
5. Next iteration recomputes everything from `residual_next`.

**No staleness:** the cache is consumed in the same iteration it was produced.
The C++ similarly stores the `two_f_B_pos` value during the parallel sweep
(line 236) and reads it back at line 301 in the same iteration. **Match.**

### (9) Stop-flag latching in C++ (first-pass noted: declared outside loop)

**Verified:** C++ at lines 178-183 declares `accept_current_loop = true;
stop_now = false; stop_none_eligible = false; stop_undone_loop = false;
stop_last_iter = false;` **outside** the for-loop body. None are reset at
iteration start. The break-protection holds because:

- `stop_none_eligible` and `stop_undone_loop` are set to True only by the
  current iteration's `if/else if` chain. They are **not** reset at iteration
  start, but they are gated by `if (stop_now) break;` at line 325, which is
  evaluated immediately after they would be set.
- `accept_current_loop` is initialized `true` and is **only flipped to `false`**
  inside `if (nEligible < 1)` (line 268) or `if (two_fs[opt_ind] > two_f_latest)`
  (line 273). Once `false`, it stays `false` (latching bug). But again, both
  flips are followed by `stop_now = true` and `break`.

**Latent risk:** If a maintainer adds a `continue;` instead of `break;` for any
of these cases, `accept_current_loop = false` persists across iterations and
silently skips the update step. This would corrupt subsequent iterations'
`x` and history. **NOT exercised today**; refactoring hazard.

The JAX `lax.scan` uses fresh per-iteration computation with `done` as the
sticky flag (line 446 `done_next = done | stop_now`). **Cleaner; not vulnerable
to the latching trap.**

### (10) JIT closure / cache

`greedy_stellarator_coil_optimization_jax` is **not jitted internally**. Callers
that wrap it in `jax.jit` provide the cache key. The function's outer level
reads:
- `n_loops = int(loops_arr.shape[0])` (line 330): static under jit (shape
  attribute).
- `n_iter = int(max_iter)` (line 331): static (must be a Python int at
  trace time).
- `no_crossing`, `no_new_coils`, `match_current`: static booleans.
- `max_loop_count_abs = abs(int(max_loop_count))` (line 335): static.

These are all baked into the trace. **JIT cache key includes them implicitly
via shape/value polymorphism.** Changing any of these triggers retracing.
**Correct behavior.**

The test `test_gsco_jax_jits_under_transfer_guard` (line 658) exercises this
under a `transfer_guard("disallow")` context to confirm no host roundtrips
occur. **Test passes; behavior verified.**

### (11) `history_length` semantics at `n_iter=0`

**Verified above.** With `n_iter=0`, both backends produce `history_length=1`
(the initial state only). The JAX `lax.scan` over `jnp.arange(0)` runs zero
iterations; carry returns initial state with `hist_ind=0`, `history_length=1`.
**Match.**

### (12) Test fixture realism

The Item 31 tests use `nGrid=5, nLoops=2` (`_gsco_problem`) — far below
production. The first-pass already noted this and the reduction-order drift
risk. Live test of moderate scale not performed in this audit (would require
constructing a full `ToroidalWireframe` and running CPP/JAX in parallel), but
the math is unchanged. **Confirm MEDIUM** — a larger fixture would expose
reduction-order drift on `f_B_history` at `rtol=1e-10`.

### (13) `bnorm_obj_matrices` ext_field / bnorm_target branches

**Live-verified above.** Both branches together produce `A_diff_max ≈ 4.7e-21`,
`b_diff_max = 0`. **Functionally correct, test coverage missing.**

---

## Untested Edge-Case Inventory (Second-Pass)

| Edge case | C++ behavior | JAX behavior | Risk |
|---|---|---|---|
| `default_current = 0` (zero increment) | `tol = 0`; all `\|x_i\| > tol` for any nonzero x; `f_S` counts all currently-active segments. `loop_curr = sign * 0 = 0`; eligibility=True always but no movement. | Same: `tol = 0`; `loop_current = directions * 0 = 0`. Eligibility passes; `delta_x = 0`; `accept_loop` stays True. The `argmin` picks index 0 (all `two_fs` equal since no move). | Both backends loop until `last_iter`; both perform no-op updates. **Wasteful but parity-preserved.** Not tested. |
| `n_loops = 1` (singleton) | `twoNLoops = 2`; `(opt_ind + 1 % 2) = opt_ind + 1`. Undo check: `(0 + 1) == 1` (positive then negative direction), works correctly. | `opt_ind + 1 == opt_ind_prev`: same check. | **Parity preserved.** Not tested. |
| All `free_loops = 0` (none free) | First iter: `nEligible = 0`, `stop_none_eligible = true`, break. | Same. | **Parity preserved (verified above).** |
| `lambda_S = inf` | `two_f = inf` for any candidate that changes `f_S`; `argmin` picks the candidate with no `f_S` change. | Same: `lambda_s * two_f_s = inf` if `two_f_s > 0`. | **Parity preserved.** Numerical edge case (`inf + finite = inf`). |
| `lambda_S < 0` (negative weight) | Adding active segments reduces `f`. Pathological but no contract check. | Same. | **Parity preserved; both backends accept invalid input.** |
| `default_current = NaN` | `tol = NaN`; `abs(x) > NaN` always False; `f_S = 0`; `loop_curr = sign * NaN = NaN`; everything propagates NaN. | Same: NaN propagates through `tol`. | **Parity preserved.** Both crash silently with NaN outputs. |
| Constraint matrix `C` with rank `< p` | `Rmat = Rtall[:p, :]` has a zero row; triangular solve → `inf` / NaN. SciPy `solve_triangular` errors loudly. | `jnp.linalg.solve(R.T, d)` returns NaN; downstream `LHS = AQ2.T @ AQ2 + WQ2.T @ WQ2` is fine but `vvec` consumes NaN. | **Different failure modes.** CPU errors, JAX produces NaN result. Coverage gap. |
| `LHS` rank < `n - p` (near-singular) | `scipy.linalg.lstsq` uses `gelsd` with `cond ≈ eps`; selects minimum-norm solution. | `jnp.linalg.lstsq(rcond=None)` uses tighter cutoff. | **DIVERGENT** (see Issue 1). |
| `max_current = 0` (zero limit) | All candidates exceed `0` since `\|x + curr\| > 0` for any nonzero curr; `n_eligible = 0`; terminate. | Same. | **Parity preserved.** |
| Very large `n_iter` with no progress | C++ loops `n_iter` times, recording the same f-values; takes O(`n_iter * n_loops * n_grid`) work. | JAX scans `n_iter` times with `done = True`; each step still computes `argmin`, scatters... — wasteful. | Both backends inefficient on no-op runs; not exposed via API hint. |
| Wireframe with shared segments across loops (e.g., adjacent cells share one toroidal segment) | `x.at[loop_inds].add(delta_x)` would update the shared segment with overlapping deltas from different selected loops across iterations. | Same. | **Parity preserved.** Standard wireframe behavior. |

---

## Recommended Actions (Deeper-Pass)

### HIGH
- **DR-1.** Mirror SciPy's `gelsd` cutoff in `regularized_constrained_least_squares_jax`
  by setting an explicit `rcond` argument: e.g.,
  `rcond = max(LHS.shape) * jnp.finfo(LHS.dtype).eps`. This eliminates the
  catastrophic-divergence regime confirmed by the live experiment.
  **File:line:** `wireframe_optimization_jax.py:149`.

### MEDIUM
- **DR-2.** Initialize `int opt_ind_prev = -1;` in C++ at
  `wireframe_optimization.cpp:154` to eliminate the UB-on-misrefactor trap.
- **DR-3.** Gate JAX `stop_undone_loop` on `~stop_none_eligible` to match C++
  mutual-exclusivity semantics at `wireframe_optimization_jax.py:406`. If the
  JAX dataclass ever exposes a `stop_reason` field, this fix is required.
- **DR-4.** Add a JAX-side rank-deficient-LHS test fixture (e.g., near-eps
  singular values) to exercise the rcond divergence regime.
- **DR-5.** Reset `accept_current_loop = true; stop_*` flags inside the C++
  for-loop body (lines 179-183) to eliminate the latching hazard.

### LOW
- **DR-6.** Add `#include <cmath>` and `#include <cstdlib>` explicitly in
  `wireframe_optimization.cpp` to remove the `abs()` resolution fragility.
- **DR-7.** Document in `WireframeGSCOResult` that consumers MUST honor
  `history_length` for trailing-zero safety.
- **DR-8.** Add explicit handling for `default_current = 0` (no-op runs)
  with an `early-return` if `default_current == 0 and x_init` already matches
  some termination criterion.

### INFO
- **DR-9.** Add a `bnorm_obj_matrices_jax` parametrized test that covers
  `(ext_field, bnorm_target, area_weighted)` × {True, False} (currently only
  the no-ext, no-target case is parametrized).
- **DR-10.** Add a large-fixture GSCO test (`nGrid ≥ 128`, `nLoops ≥ 32`) at
  relaxed `rtol=1e-9` to characterize reduction-order drift at scale.

---

## Net Verdict (Second-Pass)

The first-pass HIGH finding (`stop_undone_loop` precedence asymmetry) and the
MEDIUM findings (RCLS solver-primitive substitution, GSCO reduction-order
drift, NaN-retry omission) are corroborated by the deeper pass.

**Three NEW concerns surface from second-pass forensics:**

1. **RCLS rank-deficient LHS divergence is catastrophic, not `O(ulp · κ)`** —
   confirmed by live measurement (diff `~1.56e14` on singular values near eps).
   **Severity upgraded from INFO/MEDIUM to HIGH.**

2. **C++ `opt_ind_prev` is declared without initialization** — undefined
   behavior if reached. Safe only because of synchronized `break`. **MEDIUM
   latent trap.**

3. **JAX `stop_*` flags are not mutually exclusive** — diverges from C++
   `if/else if/else` semantics. Today this does not affect data; if any
   `stop_reason` field is later surfaced, parity breaks. **MEDIUM
   diagnostic-coverage gap.**

The JAX port remains **faithful for forward parity on the test fixtures**, with
the documented caveats. The dominant remaining risk for production use is the
RCLS divergence in the rank-deficient-LHS regime (HIGH-CONFIRMED).

