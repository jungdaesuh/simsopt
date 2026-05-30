# Transfer-Guard / lm-minpack / Residency — Test Results

Branch `gpu-purity-stage2-20260405`, atop `6cb446246`. Date 2026-05-29.
Runtime jax 0.10.0 / jaxlib 0.10.0. CPU lane = Python 3.11 local. GPU lane = A100
(CUDA, jax 0.10.0, Python 3.12 canary venv). Strict lane = `jax.transfer_guard("disallow")`.
Companion: `docs/jax_parity_status.md` (kernel/parity testing).

## 1. lm-minpack transfer-guard fix — commit `a2b4ec2c1`

`levenberg_marquardt_minpack_traceable` (`optimizer_jax.py`). Two violation
classes removed: (1) eager residual probe materialized weak host scalars
(host→device); (2) closed-over tol scalars baked via `mlir.ir_constant`
(device→host). Probe rows via `jax.eval_shape` on the pytree; build
`gradient_tol`/`gtol_value` inside `run_solver`.

| Check | Result | Lane |
|---|---|---|
| CPU repro under `disallow` (Rosenbrock → x=[1,1]) | PASS (rtol/atol 1e-10) | CPU |
| A100 toy under `disallow` (x=[1,1]) | PASS | A100 |
| A100 production `run_code(lm-minpack, residency=device)` under `disallow` | PASS (real Boozer residual, iota≈0) | A100 |
| LM / scipy-MINPACK parity (`test_lm_minpack_qr_parity` + `test_lm_damping_parity`) | 34 passed | CPU |
| Regression test `..._runs_under_transfer_guard_disallow` (added) | PASS | CPU |
| Standalone worktree @ `a2b4ec2c1` (transfer-guard + scipy-MINPACK + MGH) | 6 passed | CPU |
| ruff check / format | clean | CPU |

## 2. Linearization-residency contract — commit `b80911688`

Pins the strict-lane no-transfer invariant: device residency = identity no-op
(issues no `device_put`); host residency = device→host (CPU / SciPy-reference
lane only).

| Check | Result | Lane |
|---|---|---|
| `test_linearization_residency_device_keeps_dense_factors_unmoved` (new, object-identity) | PASS | CPU |
| Residency tests (`-k linearization_residency`: device + host + exact) | 4 passed | CPU |
| A100: `jax.device_put` GPU→CPU under `disallow` | BLOCKED (`Disallowed device-to-host transfer: (39,39), F64, cuda:0`) | A100 |
| A100: `_place_linearization_factors_for_residency(.., "host")` under `disallow` | BLOCKED (same) | A100 |
| A100 control: `np.asarray(gpu)` under `disallow` | BLOCKED | A100 |
| A100: `JAX_PLATFORMS=cuda` → `jax.devices("cpu")` | raises `Unknown backend cpu` | A100 |
| Standalone worktree @ `b80911688` (residency + lm-minpack) | 6 passed | CPU |

Note: commit initially carried a syntax error (hunk misplacement under `-U0`
`--unidiff-zero`); caught via committed-blob `py_compile`, fixed by `--amend`;
committed file compiles.

## 3. Single-stage lm-minpack E2E (A100, scipy-jax outer, cuda, `disallow`)

| Run | Tree | maxiter | rc | wall | transfer crashes | outcome |
|---|---|---|---|---|---|---|
| baseline (prior) | snapshot 13a664f15, **unfixed** optimizer | 8 | — | — | **1** (`Disallowed device-to-host transfer`) | crash at inner lm-minpack solve |
| run 1 | fixed `src/src` | 3 | 1 | 138s | **0** | blocked at `make_traceable` (before inner solve) |
| run 2 | snapshot + fix | 2 | 1 | 134s | **0** | same `make_traceable` block (both trees) |

The fix is **exonerated** — zero transfer crashes once loaded. The `rc=1` is a
separate, pre-existing gate blocker (§4), not the lm-minpack fix.

## 4. `make_traceable` gate — root cause + codex fix

Root cause (clean-HEAD trace): the gate compares the resolved inner-solve
**method** string against a stale allow-set
`{"bfgs-ondevice","lbfgs-ondevice","lm-ondevice"}` that omits
`"lm-minpack-ondevice"` / `"optimistix-lm-ondevice"`. The backend IS `ondevice`
and propagates correctly; the error string (`requires optimizer_backend='ondevice'`)
is a misnomer. Default `lm-ondevice` runs are unaffected. HEAD location
`surfaceobjectives_jax.py:4052`; codex moved it to `surfaceobjectives_traceable_jax.py:1113`.

codex fix (SSOT) — verified:

| Check | Result |
|---|---|
| Gate references `_ONDEVICE_OPTIMIZER_METHODS` (SSOT) | yes |
| `py_compile` | OK |
| Module imports (no cycle); gate set `is` the SSOT object | True |
| Resolved allow-set | `{bfgs-ondevice, lbfgs-ondevice, lm-minpack-ondevice, lm-ondevice, optimistix-lm-ondevice}` |
| `lm-minpack-ondevice` ∈ set | True |
| Sibling `run_code_traceable` gate (`boozersurface_jax.py:5349`) | already SSOT-consistent |

## 5. Deferred / not run

- Full single-stage lm-minpack E2E on GPU (outer → gate → inner solve, composed):
  every link verified independently; a clean run is best done **after** codex
  commits item-14 (push → clone → run). Not a blocker for the committed fixes.
- codex's gate fix lives in an **untracked** file (`surfaceobjectives_traceable_jax.py`,
  lands with item-14) — not yet committed.

## 6. Commits

- `a2b4ec2c1` — fix: keep lm-minpack-ondevice safe under transfer_guard("disallow")
- `b80911688` — docs(jax): pin strict-CUDA linearization-residency no-transfer contract
