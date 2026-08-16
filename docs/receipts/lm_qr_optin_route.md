# Receipt — LM_QR opt-in dense-QR Levenberg-Marquardt route

> **Status: diagnostic, not certifying.** Quiet-box single-run timings on a
> shared workstation; no contention control, no repeat-across-boot, no
> SHA-pinned artifact bundle. Use for direction, not for promotion claims.

- **Date:** 2026-08-16
- **Commit:** `20ef68f7be82dab8b9cb106e49819c13908f96cf` (`pr/jax-port-squashed`)
- **Box:** AMD Ryzen Threadripper 9970X (64 threads), CPU-only lane
- **Runtime:** JAX 0.10.0, Python 3.11.15, `JAX_ENABLE_X64=1`, `JAX_PLATFORMS=cpu`,
  `CUDA_VISIBLE_DEVICES=` (empty)
- **Working tree:** *not* clean at measurement time — sibling change-sets from
  the same integration wave were in flight (`src/simsopt/`, `examples/`,
  `tests/`; a later provenance-pin refresh also touched
  `src/simsopt_jax/objectives/single_stage_fullspace.py`). Those paths are
  **disjoint** from the LM route: `git diff -- src/simsopt_jax/geo/optimizers/
  src/simsopt_jax/solve/ src/simsopt_jax_adapters/` was **empty**. Recorded
  here rather than hidden, because a receipt taken on a dirty tree cannot be
  promoted to certifying without a clean-tree re-run.

---

## 1. What the route is

`Driver.SIMSOPT_LM_QR` is an **opt-in** on-device Levenberg-Marquardt lane whose
inner linear solve is a dense, column-pivoted QR factorization of the *damped
augmented* system — DESC-style factorize-once — instead of the matrix-free GMRES
used by the default lane.

Per LM iteration it forms

```
[    J     ]        [ -r ]
[ sqrt(l)I ] dx  =  [  0 ]
```

and solves it by column-pivoted QR + triangular back-substitution, then undoes
the pivot permutation.

**Implementation:** `src/simsopt_jax/geo/optimizers/optimizer.py`

| what | where |
|---|---|
| Solver entry | `levenberg_marquardt_minpack_traceable` — `optimizer.py:3885` |
| Augmented-system QR step | `_qr_lm_step` — `optimizer.py:3734` |
| Dense state (J, g, H, cost) | `_qr_lm_dense_state` — `optimizer.py:3657` |
| QR-scaled gradient norm (MINPACK `info=4`) | `_qr_scaled_gradient_norm` — `optimizer.py:3678` |
| Outer loop | `_qr_lm_iteration` — `optimizer.py:3761` |

**Why the augmented form rather than the normal equations.** Solving
`(J^T J + lambda I) dx = -J^T r` explicitly forms `J^T J`, which squares the
condition number (`kappa(J^T J) = kappa(J)^2`). Factorizing `[J; sqrt(lambda)I]`
directly keeps the solve conditioned at `kappa(J)`, so it retains roughly twice
the significant digits on the ill-conditioned surface problems this repo
targets. Column pivoting additionally handles rank-deficient `J`, which is the
regime the Marquardt damping exists to survive. This is the standard MINPACK
`lmpar` conditioning model — hence the `lm-minpack` spelling — though this lane
does **not** claim MINPACK packed-QR byte identity.

---

## 2. How it is selected (explicit only — never a default)

Three equivalent opt-in surfaces, all requiring an explicit string:

| surface | value |
|---|---|
| Typed driver | `Driver.SIMSOPT_LM_QR` + `SimsoptLMQROptions` |
| Boozer routing contract | `least_squares_algorithm="lm-minpack"` (`optimizer.py:1304`) |
| Legacy method string | `method="lm-minpack-ondevice"` (`driver.py:57`) |

**Nothing reaches this lane by default.**
`resolve_boozer_least_squares_algorithm` (`single_stage_routing.py:143`) returns
only `"lm"` or `"quasi-newton"` when the caller passes nothing, for every
supported backend. `"lm"` continues to resolve to `Driver.SIMSOPT_LM_GMRES`
(`optimizer.py:1299`). `"lm-minpack"` appears in
`VALID_LEAST_SQUARES_ALGORITHMS` (`optimizer.py:494`) as a *choice*, never as a
default value.

**Route-string visibility (so receipts can pin the lane):** the result records
the executed route — `result.driver is Driver.SIMSOPT_LM_QR`, whose `.value` is
the literal string `simsopt_lm_qr`; the legacy spelling is recoverable via
`legacy_target_least_squares_method(Driver.SIMSOPT_LM_QR) == "lm-minpack-ondevice"`.

---

## 3. Dense-materialization cap

The QR lane materializes a dense Jacobian, so it is gated **before** any
materialization happens. `levenberg_marquardt_minpack_traceable` probes the
residual row count with `jax.eval_shape` (traces without executing, and without
tripping `transfer_guard`), then consults the shared policy helper and fails
closed:

- policy: `_least_squares_dense_linearization_policy` — `optimizer.py:4465`
- refusal: `MemoryError` raised at `optimizer.py:3937-3944`
- message builder: `_least_squares_required_dense_linearization_message` — `optimizer.py:4445`

The budget reuses the **existing repo-wide convention**, `max_dense_linearization_bytes`
(SSOT — same field name and same `J + H` byte accounting already used by
`SimsoptLMGMRESOptions` and `OptimistixLMOptions`), declared at
`solve/simsopt/contracts.py:75` and threaded through `solve/dispatch.py:192`.
It is `None` (unset) by default on all three options classes: the caller
declares the budget it is willing to spend.

Observed refusal message (40x8 fixture, float64 — `40*8*8 + 8*8*8 = 3072` bytes):

```
Levenberg-Marquardt dense QR solve requires residual Jacobian/Hessian artifacts
totaling 3072 bytes in dtype float64, exceeding max_dense_linearization_bytes=1.
```

---

## 4. Byte-oracle status — defaults untouched

**No source file was modified for this work.** `git diff --
src/simsopt_jax/geo/optimizers/ src/simsopt_jax/solve/
src/simsopt_jax_adapters/` is empty at the measured commit, so every default is
byte-identical by construction, and every SHA-bound receipt pinning
`simsopt_lm_gmres` (strict-results JSONL, TDD receipt, progress report) remains
valid without re-issue.

The `lm` <-> `lm-ondevice` (host-GMRES <-> on-device-GMRES) byte-equality oracle
is **preserved**: the QR lane is a third, separately-named driver and does not
sit on that pair's path.

Existing LM/least-squares routing suites, run unmodified, one file per process
(`CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu JAX_ENABLE_X64=1`):

| file | result |
|---|---|
| `tests/jax/solve/test_driver_dispatch.py` | 9 passed |
| `tests/jax/solve/test_compat_shim_translation.py` | 13 passed |
| `tests/jax/solve/test_optimizer_result_schema.py` | 4 passed |
| `tests/solve/test_serial_jax.py` | 21 passed |
| `tests/geo/test_optimizer_jax_item19.py` | 21 passed |
| **total** | **68 passed, 0 failed, 0 expectations edited** |

New contract suite `tests/jax/solve/test_lm_qr_optin_route.py`: **14 passed**
(routing-never-default pins, route-string visibility, optimum agreement on a
linear and a nonlinear fixture, explicit not-byte-equal pin, cap refusal +
message content + within-budget success, shared-budget-convention pin).

**Why the QR lane is deliberately not byte-compared to GMRES.** The two lanes
solve different algebra: GMRES iterates matrix-free on `(J^T J + lambda I)`,
QR factorizes `[J; sqrt(lambda)I]`. Different rounding and a different accepted-step
trajectory follow necessarily — on `exp_fit_60x3` the lanes take 17 vs 18
iterations. They are pinned as **tolerance-equivalent at the optimum**
(`atol=1e-7`), and separately pinned as **not** bit-identical, so that an
accidental aliasing of the two routes would fail the suite.

---

## 5. Measured fixture comparison — `lm_qr` vs `lm_gmres`

> **SUPERSEDED by Addendum A (2026-08-16).** The numbers below are pre-fix and
> measure a per-call retrace, not dense-QR algebra. Read Addendum A first.

Three fixtures, `maxiter=400`, median of 5 timed repeats after one warm-up call.
"cold" = first call (trace + compile + execute); "warm" = median subsequent call.

### 5a. As shipped (no persistent compilation cache)

| fixture | shape | route | iters | cold (s) | warm (ms) | final cost |
|---|---|---|---|---|---|---|
| `linear_ls_40x8` | 40 x 8 linear | `lm_qr` | 2 | 0.17 | 128.86 | 7.996562237100e+00 |
| | | `lm_gmres` | 2 | 0.23 | **0.15** | 7.996562237100e+00 |
| `exp_fit_60x3` | 60 x 3 nonlinear | `lm_qr` | 17 | 0.16 | 138.11 | 8.45e-19 |
| | | `lm_gmres` | 18 | 0.27 | **0.23** | 1.23e-23 |
| `linear_ls_400x60` | 400 x 60 linear | `lm_qr` | 2 | 0.18 | 162.99 | — |
| | | `lm_gmres` | 2 | 0.26 | **11.56** | — |

Cross-route agreement at the optimum (max abs component difference):
`linear_ls_40x8` 2.78e-17 · `exp_fit_60x3` 8.67e-10 · `linear_ls_400x60` 1.17e-15.
All three well inside the `atol=1e-7` pin; none bit-identical, as expected.

### 5b. Same run with the persistent compilation cache enabled

| fixture | `lm_qr` warm (ms) | `lm_gmres` warm (ms) | ratio |
|---|---|---|---|
| `linear_ls_40x8` | 52.81 | 0.17 | 306x |
| `exp_fit_60x3` | 50.34 | 0.39 | 129x |
| `linear_ls_400x60` | 58.86 | 11.61 | 5.1x |

### 5c. Reading the numbers — the ratio is a caching artifact, not QR algebra

The headline ratios (5.1x - 865x) must **not** be read as "dense QR is ~800x
slower than GMRES". The QR lane's per-call wall is dominated by *re-tracing and
re-compiling on every invocation*. Two independent lines of evidence:

1. **Flat vs. scaling.** Across a 50x problem-size increase (40x8 -> 400x60),
   the QR per-call wall moves only 52.81 -> 58.86 ms (**+11%**), while GMRES
   scales 0.17 -> 11.61 ms (**68x**). Real algebra scales with problem size;
   compile time does not.
2. **Cache sensitivity.** Enabling the persistent compilation cache cuts QR warm
   time by ~2.6x (129-163 ms -> 50-59 ms) while barely moving GMRES. Only
   compilation responds that way.

**Root cause (open finding, not fixed here).**
`src/simsopt_jax/geo/optimizers/optimizer.py:4031`:

```python
state = jax.jit(run_solver)(flat_x0)
```

`run_solver` is a **local closure rebuilt on every call**, so `jax.jit` keys a
fresh cache entry each invocation and never hits its cache. The comparable
lanes avoid this by returning `jax.jit(run_solver)` from a *memoized builder*
routed through `_cached_private_solver` — see `optimizer.py:8061`, `:8713`,
`:9200`. Wiring the QR lane onto that same seam should remove essentially all
of the observed gap, at which point the honest QR-vs-GMRES algebra comparison
can be measured for the first time.

This was left unlanded deliberately: it is a caching/Tier-1 change to a solver
seam with known hazards (the `_cached_private_solver` marker must survive
`functools.wraps`, and cache keys must carry the closure-constants signature),
and it deserves its own review rather than riding along inside a receipt.

**What the measurement does establish:** both lanes converge, to the same
optimum, with comparable iteration counts (2 vs 2, 17 vs 18, 2 vs 2). The QR
lane is functionally correct; its current cost is an unresolved compilation
issue, not a numerical one.

---

## 6. Reproduce

```bash
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
    PYTHONPATH=src .venv-qn-gpu/bin/python \
    -m pytest tests/jax/solve/test_lm_qr_optin_route.py -q
```

Timings in section 5 come from a standalone harness driving
`simsopt_jax.solve.dispatch.least_squares` over the two drivers with the
fixtures defined in that test file (`_linear_fixture`, `_nonlinear_fixture`) plus
a 400x60 linear case built the same way (`numpy` default_rng seed `20260816`).

---

## Addendum A — per-call retrace fixed; first honest QR-vs-GMRES algebra number (2026-08-16)

> **Supersedes the "Root cause (open finding, not fixed here)" paragraph in
> section 5c.** The retrace is landed as fixed; sections 5a/5b above are now
> *historical* pre-fix numbers and must not be quoted as the cost of dense QR.
>
> **Status: still diagnostic, not certifying.** Same quiet-box, single-box,
> no-repeat-across-boot caveats as the header.

- **Date:** 2026-08-16
- **Base commit:** `0cc16903109cb355ffa90de402bb74c5a87771d3` (`pr/jax-port-squashed`)
- **Box / runtime:** identical to the header (Threadripper 9970X, 64 threads,
  JAX 0.10.0, Python 3.11.15, `JAX_ENABLE_X64=1`, `JAX_PLATFORMS=cpu`,
  `CUDA_VISIBLE_DEVICES=` empty)
- **Working tree:** the LM route carried this addendum's own uncommitted change
  (`src/simsopt_jax/geo/optimizers/optimizer.py`,
  `tests/jax/solve/test_lm_qr_optin_route.py`). Unrelated in-flight edits from a
  concurrent integration wave were present in `docs/` and are disjoint from the
  solver path. Recorded rather than hidden, per the header's rule.

### A.1 The change

The QR lane now routes through the same memoized runner seam the GMRES lane
uses. Previously `levenberg_marquardt_minpack_traceable` closed over
`residual_fn`, `args`, `unravel`, the tolerance scalars and the callbacks in a
*locally rebuilt* `run_solver`, then called `jax.jit(run_solver)(flat_x0)` — a
fresh Python function object per call, so the JIT cache was never hit.

| what | where |
|---|---|
| Memoized entry (cache key = build-time constant set) | `_make_traceable_levenberg_marquardt_minpack_runner` — `optimizer.py:3886` |
| Runner builder (returns the jitted program) | `_build_traceable_levenberg_marquardt_minpack_runner` — `optimizer.py:3929` |
| Shared seam (weakref cache entry, lock-held build) | `_cached_traceable_runner` — `optimizer.py:1149` |
| Per-lane cache dict | `_TRACEABLE_LM_QR_RUNNER_CACHE` — `optimizer.py:569` |
| Traced callback-token operand (both LM lanes) | `_traceable_callback_token_operand` — `optimizer.py:1226` |

**Scope beyond the opt-in lane, disclosed up front.** Section 4's "no source
file was modified" no longer holds for this addendum, and one of the touched
lanes is the **default** one: A.7's token fix edits the `lm_gmres` builder and
entry point as well. That lane is a byte-equality oracle partner, so its edit
carries its own pre/post bitwise A/B (A.7) rather than riding on the QR lane's.
No dispatch default, routing rule, or option default is changed by either edit.

What used to be a baked-in closure constant is now either a runtime argument or
a cache-key component, so no two problems can share a wrong executable:

- **residual callable** — owns the cache entry, keyed by object identity behind
  a weakref. The dead-entry reaping is real but **conditional**: it applies to
  bare identity-keyed callables, which get a weakref finaliser that drops the
  entry when the callable dies. A callable carrying an explicit
  `_TRACEABLE_RUNNER_CACHE_TOKEN_ATTR` is keyed by that semantic token instead
  and deliberately gets *no* finaliser, so its runners persist for the process
  lifetime — bounded by the number of distinct tokens, not by call count. The
  production Boozer residual reaches the lane through the token-marked path via
  `_get_traceable_penalty_residual`'s own per-option-key memo, so its retention
  is bounded by that option-key count.
- **`maxiter`, `tol`, `ftol`, `xtol`, `gtol`, callback-enabled,
  progress-callback-enabled** — cache key. These are the *only* values the
  builder bakes into the trace, so the key is exactly the builder's signature.
- **`args`** — promoted to a runtime `fn_args` argument (the GMRES lane's
  convention). Baking them would have been the silent-wrong-answer failure: two
  solves with different residual data and identical shapes would have replayed
  the first answer. Pinned by
  `test_qr_lane_reuses_one_executable_across_residual_arg_values`.
  **Deliberate domain narrowing, disclosed:** as a closure constant, `args`
  could hold *arbitrary* Python objects; as a JIT argument every entry must now
  be a JAX-typed value or a registered pytree, and a non-pytree object raises at
  the call instead of being captured silently. This is accepted rather than
  worked around because (a) the default GMRES lane already imposed exactly this
  contract, so the narrowing aligns the two lanes rather than splitting them,
  and (b) the sole in-repo caller — `boozer_surface.py:6563`, `args=(coil_set_spec,)`
  — passes a `GroupedCoilSetSpec`, registered as a pytree at
  `core/specs.py:481`, and already feeds the same object to the GMRES lane.
- **`unravel`** — moved *inside* the trace (`ravel_pytree(x_init)` on the traced
  pytree). The decision-vector structure therefore never enters the runner cache
  key at all: `jax.jit`'s own argument signature discriminates it. For a flat
  `x0` the in-trace ravel is a no-op reshape/concatenate of one operand, which
  is why the traced program is unchanged (see A.2).
- **`callback` / `progress_callback`** — moved onto the existing traceable
  callback-token registry, with a flat-vector adapter that restores the caller's
  pytree before the user callback sees it. The tokens are **traced int32
  operands**, staged by `_traceable_callback_token_operand`
  (`optimizer.py:1226`), so an instrumented solve reuses the same executable as
  every other instrumented solve and the lane's compiled-program count stays at
  one. See A.7 — the first cut of this addendum declared them `static_argnums`,
  which was a retention leak, and fixing it also repaired the same pre-existing
  defect in the default GMRES lane.

The dense-materialization cap of section 3 is unchanged and still fails closed
*before* materialization, on the host, ahead of the runner call.

**Why this seam and not `_cached_private_solver`.** Section 5c pointed at
`_cached_private_solver` as the comparable-lane seam. It is not the one the LM
family uses: that helper lives in `private/_common.py:238`, serves the scalar
BFGS/L-BFGS lanes, and is gated on the `_CACHEABLE_VALUE_AND_GRAD_ATTR` marker
that `mark_cacheable_jit_value_and_grad` writes onto *value-and-gradient
objective* callables. Least-squares **residual** callables never carry *that*
marker — including the production one, `boozer_surface.py:6347`
`_get_traceable_penalty_residual` — so routing the QR lane through it would have
returned `builder()` uncached on every call and fixed nothing. (That residual is
marked, but with the different `_TRACEABLE_RUNNER_CACHE_TOKEN_ATTR` marker of
the traceable-runner seam — `boozer_surface.py:6377`,
`cache_token=("boozer-traceable-penalty-residual", key)` — which is exactly the
marker the seam adopted here reads.) The seam the
sibling LM lane actually uses is `_cached_traceable_runner`, which keys on the
callable itself, and that is what this change adopts. Because
`_get_traceable_penalty_residual` memoizes its closure per option key, the
production Boozer path returns the same callable object across calls and so
benefits from the fix, not only the fixtures.

### A.2 Bitwise contract — proof

**Method.** A fixture battery was run through `simsopt_jax.solve.dispatch.least_squares`
against the tree at `0cc169031` *before any edit* (`git diff -- src/ tests/`
empty at capture time, so this is exactly the pre-change lane), dumping every
`OptimizerResult` field as an exact digest: SHA-256 over the array's raw bytes,
plus its first six components as C99 hex floats, and `float.hex()` for scalars.
The identical harness was re-run after the change and the two JSON captures
compared field by field. No toggle, no tolerance.

**Battery (16 records, 13 of them QR-lane).** `linear_ls_40x8`, `exp_fit_60x3`,
`linear_ls_400x60`; the first two also re-solved warm to prove the *cached*
executable — not just the first one — is byte-equal. Plus four paths the change
specifically touches: a `residual_args` fixture solved with two different arg
values and then the first value again (runtime-args path), a dict-structured
`x0` (in-trace `ravel_pytree` path), a callback + progress-callback run
capturing all nine delivered step payloads (token-registry path), and a
`maxiter=3` early-stop run (non-converged `status`/`success`/`nit` fields).

**Result: 16/16 byte-identical, 0 mismatches** across `x`, `fun`, `jac`, `nit`,
`nfev`, `njev`, `status`, `success`, `message`, `residual`, `residual_jacobian`,
`hessian`, and the callback payload stream.

**Excluded from the comparison, and why:** `OptimizerResult.wallclock_s`, which
`dispatch.least_squares` stamps from `time.perf_counter()` and which therefore
differs on every run of *any* build — it is a measurement, not a result. It is
the only result field left out; `driver` and `options_used` are echoes of the
call and are pinned separately by section 2's route-string tests. Making
`wallclock_s` differ is in fact the whole point of the change, and A.3 reports
that difference rather than hiding it.

Spot values (unchanged pre/post):

| record | `nit` | `fun` (hex float64) | SHA-256(`x`) prefix |
|---|---|---|---|
| `linear_ls_40x8.qr` | 2 | `0x1.ffc7acfa3177dp+2` | `c49f7cf682c11db7` |
| `linear_ls_40x8.qr` (warm) | 2 | `0x1.ffc7acfa3177dp+2` | `c49f7cf682c11db7` |
| `exp_fit_60x3.qr` | 17 | `0x1.f2f9787b58b80p-61` | `995aa8df1f93ee95` |
| `linear_ls_400x60.qr` | 2 | `0x1.5dc7ed776f9dap+7` | `4337f4e4e4a9a426` |
| `args_30x5.qr_a` | 2 | `0x1.7c7469ed07273p+3` | `da550d36f04bd462` |
| `args_30x5.qr_b` | 2 | `0x1.d3310964d4c04p+2` | `a55f4a21a5ee1753` |
| `args_30x5.qr_a_again` | 2 | `0x1.7c7469ed07273p+3` | `da550d36f04bd462` |

The GMRES lane's records in the same battery are byte-identical too, as required
by section 4 (`linear_ls_40x8.gmres` `0x1.ffc7acfa3177cp+2`, SHA `5eb3d8a1…`).

### A.3 Re-measured fixtures — post-fix

Same methodology as section 5: `maxiter=400`, one warm-up call, median of 5
timed repeats, no persistent compilation cache. Pre-fix column is a *fresh*
re-run of section 5a on this box today, so both columns are same-box, same-day.

| fixture | shape | route | iters | cold (s) | warm pre-fix (ms) | warm post-fix (ms) | speedup |
|---|---|---|---|---|---|---|---|
| `linear_ls_40x8` | 40 x 8 linear | `lm_qr` | 2 | 0.16 | 129.81 | **0.485** | 268x |
| | | `lm_gmres` | 2 | 0.27 | 0.173 | 0.141 | — |
| `exp_fit_60x3` | 60 x 3 nonlinear | `lm_qr` | 17 | 0.15 | 136.42 | **0.703** | 194x |
| | | `lm_gmres` | 18 | 0.27 | 0.234 | 0.227 | — |
| `linear_ls_400x60` | 400 x 60 linear | `lm_qr` | 2 | 0.24 | 158.62 | **3.264** | 49x |
| | | `lm_gmres` | 2 | 0.28 | 11.350 | 12.349 | — |

Iteration counts and final costs are unchanged from section 5 (2/2, 17/18, 2/2),
as A.2 requires. A second full repeat of the harness reproduced every QR cell
within 4% (0.505 / 0.690 / 3.335 ms). The GMRES repeat was 0.185 / 0.237 /
12.425 ms: the 400 x 60 cell within 1%, but the two sub-0.25 ms cells moved up
to 31% between runs, which is the dispatch-latency floor showing rather than a
change in the lane. Sub-millisecond cells in either lane should be read as
order-of-magnitude, not as three significant figures.

**End-to-end warm ratio, post-fix:** QR is 3.4x slower at 40x8, 3.1x slower at
60x3, and **3.8x faster** at 400x60. The 5.1x-865x headline of section 5 is gone.

### A.4 The algebra verdict

The end-to-end wall still mixes device algebra with per-call host work, so the
compiled programs were timed directly (median of 5, warm, calling the memoized
runners; the QR lane's `jax.eval_shape` cap probe timed separately):

| fixture | host cap probe (ms) | `lm_qr` exec (ms) | `lm_gmres` exec (ms) | `lm_gmres` matrix-free exec (ms) |
|---|---|---|---|---|
| `linear_ls_40x8` | 0.241 | **0.046** | 0.072 | 0.074 |
| `exp_fit_60x3` | 0.385 | **0.088** | 0.128 | 0.086 |
| `linear_ls_400x60` | 0.198 | **2.444** | 12.024 | 11.201 |

("matrix-free" = GMRES with `materialize_dense_linearization=False`, i.e. no
post-hoc dense Jacobian/Hessian at all — the configuration most favourable to
GMRES, and the conservative comparator for a QR-wins claim.)

**Verdict: dense column-pivoted QR of the augmented system is not the cost
driver. On the one fixture large enough to measure — 400 x 60 — the QR program
is 4.9x faster than the matched-work GMRES program and 4.6x faster than
matrix-free GMRES.** The two small fixtures land at 1.5x / 1.5x (matched work)
and 1.6x / 1.0x (matrix-free), but at 46-128 us they sit at the
dispatch-latency floor and should be read as "no measurable difference", not as
two further wins. The load-bearing number is the 400 x 60 one; it is not a
scaling *curve* (one wide point, not a sweep), and calling it one would need the
sweep this addendum does not have.

This is the number the campaign was missing, and it points the opposite way from
section 5's headline. Factorize-once QR beats matrix-free GMRES on the inner
solve as soon as the column count is non-trivial: over the 8 -> 60 column
increase the GMRES program goes 0.074 -> 11.2 ms (151x) while the QR program
goes 0.046 -> 2.4 ms (53x). The mechanism is visible in the GMRES step itself
(`_gmres_solve_least_squares_system`, `optimizer.py:2703`), whose Krylov budget
is sized from `n`: `restart` 8 -> 50 and `maxiter` 32 -> 200 across those two
fixtures, on top of a more expensive matvec.

**The remaining end-to-end deficit at small size is host preflight, not algebra.**
The QR lane pays a per-call `jax.eval_shape` probe (0.20-0.39 ms) to establish
the residual row count *before* any dense materialization, so that the section-3
byte cap can fail closed. The GMRES lane pays nothing comparable: it computes
rows/cols from concrete shapes *inside* its runner, because it has no up-front
materialization to refuse. On `linear_ls_40x8` that fixed probe is ~5x the
entire device solve. This is now the largest remaining QR-lane overhead and is
left as an explicit open finding — collapsing it (e.g. memoizing the probe on
the same runner-cache key, or deriving rows from the runner's own output when no
budget is declared) is a separate change with its own fail-closed argument to
make, and it must not ride along inside a receipt.

### A.5 Suites run

One file per process, `CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu JAX_ENABLE_X64=1`.

| file | result |
|---|---|
| `tests/jax/solve/test_driver_dispatch.py` | 9 passed |
| `tests/jax/solve/test_compat_shim_translation.py` | 13 passed |
| `tests/jax/solve/test_optimizer_result_schema.py` | 4 passed |
| `tests/solve/test_serial_jax.py` | 21 passed |
| `tests/geo/test_optimizer_jax_item19.py` | 21 passed |
| **section-4 sweep total** | **68 passed, 0 expectations edited** |
| `tests/jax/solve/test_lm_qr_optin_route.py` | 25 passed (14 pre-existing, unedited + 11 new) |
| `tests/geo/test_boozersurface_jax.py` (supplementary) | 533 passed, 25 skipped |
| `tests/geo/test_surface_objectives_jax.py` (supplementary) | 331 passed, 27 skipped |

Every row was re-run after the A.7 token fix, because that fix touches the
default GMRES lane and a pre-fix suite result is not evidence for post-fix code.

**One known-red gate, by construction, not fixed here.**
`tests/benchmarks/test_rehearse_single_stage_projected_route_cpu.py::test_execution_source_binding_accepts_the_live_repository`
fails with `module simsopt_jax.geo.optimizers.optimizer executes
src/simsopt_jax/geo/optimizers/optimizer.py with bytes the manifest does not
describe`. That is the execution-source manifest doing its job: it pins the
byte digest of every executed module, and this addendum edits one of them.
Refreshing `benchmarks/single_stage_native_equivalent_quality_gntr3_execution_sources.json`
is an integration step reserved to whoever lands the change, deliberately not
performed here — a receipt must not re-mint the authority that audits it. The
gate does not self-heal: it reads the manifest and refuses, leaving the file
untouched.

The 11 new cache-contract tests assert reuse by **retrace counting**, not by wall
time: the residual callable is wrapped in a counter, and because a jitted body
runs its Python only while tracing, a warm solve must cost exactly the one
preflight residual trace (the cap probe of A.4) while a cold solve costs more.
They cover warm reuse, runner-object memoization plus per-constant-set
separation, two residuals with different embedded data keeping separate programs
(and the first staying warm afterwards), runtime `args` reuse-without-staleness,
structured-`x0` reuse, two decision structures staying apart inside one runner,
callback pytree delivery through the token registry, the A.7 instrumented-solve
cache-flatness pin on **both** LM lanes, and
warm reuse under **both** `SIMSOPT_TARGET_LANE_STRICT=0` and `=1`. The strict
flag is inert here by inspection —
`wrap_strict_target_lane_value_and_grad` (`optimizer.py:1586`) is applied only
in the scalar `minimize` entrypoints, never on the least-squares residual path —
and the parametrized test pins that, because a strict-mode-only wrapper object
interposed in front of the residual is exactly what would silently defeat a
callable-identity cache key.

### A.6 Reproduce

Suites: the section-6 command, unchanged.

A.3 timings: the section-6 harness recipe, unchanged — `least_squares` over both
drivers, `maxiter=400`, one warm-up then the median of 5, with `_linear_fixture`
and `_nonlinear_fixture` from the test file plus a 400 x 60 linear case built the
same way (`numpy` `default_rng` seed `20260816`).

A.4 executable-level timings: same fixtures and same median-of-5 protocol, but
calling the memoized runners directly to exclude host preflight and result
plumbing —

```python
qr = _make_traceable_levenberg_marquardt_minpack_runner(
    residual, 400, 1e-10, 1e-8, 1e-8, 1e-8, False, False)
gmres = _make_traceable_levenberg_marquardt_runner(
    residual, 400, 1e-10, 1e-8, 1e-8, None, True, None, False, False)
# ...and the same gmres builder with materialize_dense_linearization=False
# for the matrix-free column.
state = qr(x0, ()); jax.block_until_ready(state["x"])
```

The probe column times
`jax.eval_shape(lambda p: jnp.ravel(jnp.asarray(residual(p))), x0)` alone.

A.2 bitwise capture: run the fixture battery through
`simsopt_jax.solve.dispatch.least_squares` and dump, per result field,
`hashlib.sha256(np.ascontiguousarray(field).view(np.uint8)).hexdigest()` plus
`float.hex()` of the first six components; do it once on a clean checkout of the
base commit and once on the changed tree, then compare the two dumps for
equality.

A.7 retention measurement: solve `exp_fit_60x3` 100 times through
`dispatch.least_squares` with a `callback=` attached, sampling
`resource.getrusage(RUSAGE_SELF).ru_maxrss` and the lane runner's
`_cache_size()` every 20 solves. The runner handle comes from the same two
builder calls as A.4, with both callback flags set to `True`.

### A.7 Correction — the instrumented path retained one executable per solve

The first cut of this addendum kept the callback tokens as `static_argnums`,
reasoning that instrumented solves would then "compile per call by construction,
the same trade the GMRES lane makes". That reasoning was wrong in a way the
memoization made newly harmful, and a delta audit caught it.

**The defect.** Tokens are minted per call, so a static token compiles a fresh
executable per instrumented solve. Before memoization the jitted object was a
per-call local and was freed with it. Once the runner is memoized, the lane
*retains* every one of those executables for the process lifetime, with no cap.
Measured over 100 callback-observed `exp_fit_60x3` solves:

| lane | compiled programs retained | RSS start -> end | per solve |
|---|---|---|---|
| `lm_qr`, static tokens | 100 | 567 -> 1334 MB | ~7.7 MB |
| `lm_gmres`, static tokens | 100 | 567 -> 1755 MB | ~11.9 MB |
| `lm_qr`, traced tokens | **1** | 567 -> 679 MB (flat from solve 20) | ~0 |
| `lm_gmres`, traced tokens | **1** | 564 -> 695 MB (flat from solve 20) | ~0 |

Callback delivery is unaffected in every row (900/900 for QR, 1000/1000 for
GMRES).

**Scope note, stated plainly:** the `lm_gmres` rows are a **pre-existing** defect
of the default lane, not something this change introduced — that lane was
already memoized and already used static tokens at the base commit. The QR rows
*were* introduced by this change and are now fixed. Both are repaired here.

**The fix.** Drop `static_argnums` on both LM builders and stage the tokens as
traced `int32` operands via `_traceable_callback_token_operand`
(`optimizer.py:1226`), which uses the repo's explicit device-placement helper
rather than an implicit transfer. This is sound because the token never touches
solver numerics: it is consumed only by `_lookup_traceable_callback`, which
already does `int(np.asarray(token).reshape(()).item())` and so accepts arrays.

**Bitwise proof for the default lane.** The GMRES lane is a byte-oracle partner
(section 4), so its edit was A/B'd on its own: five instrumented records
captured before and after — `exp_fit_60x3` and `linear_ls_40x8` under
`lm_gmres`, `exp_fit_60x3` under `lm_qr`, plus a repeat solve of each lane to
prove the *shared* executable reproduces the stream. All five are identical in
both the full result-field digest set and the complete callback payload stream
(iteration, `fun`, `grad_norm_inf`, and `x` digest per event; 10/10, 2/2, 9/9,
10/10, 9/9 events). The 16-record A.2 battery was then re-run and remains
16/16 identical to the base commit.

**Regression pin.** `test_instrumented_solves_share_one_compiled_executable`
runs 12 instrumented solves per lane and asserts the lane retains exactly one
compiled program for that residual, summed across every memoized runner filed
under it — so it catches both a forked JIT cache and surplus runner objects. It
is parametrized over `lm_qr` and `lm_gmres`, and it fails on the pre-fix code
with `12 == 1` on **both** lanes.

**Adjacent, out of scope, reported not fixed:** the traceable Newton-polish
builder at `optimizer.py:7193-7194` uses the same `static_argnums` token idiom
for its progress-callback token — while its *matvec-counter* token, three
branches earlier at `:7182`, is already passed traced. That asymmetry inside one
builder is a hint the static spelling is incidental rather than required, but the
lane was not touched here — it is outside this receipt's ownership and needs its
own bitwise
A/B — but it carries the same signature and should be assessed separately.
