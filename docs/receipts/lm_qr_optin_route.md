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
