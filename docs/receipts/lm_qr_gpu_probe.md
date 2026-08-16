# Receipt — LM_QR dense-QR lane on GPU: correctness probe and width ladder

> **Status: diagnostic, not certifying.** Single box, single boot, quiet-gated
> but not contention-controlled beyond that gate; no SHA-pinned artifact bundle;
> no repeat-across-boot. Use for direction, not for promotion claims. In
> particular, **no number here may be quoted as a certified GPU win.**

- **Date:** 2026-08-16
- **Commit:** `829d92f23` (`pr/jax-port-squashed`). **No `src/` or `tests/` file
  was modified by any phase of this campaign** (`git diff -- src/ tests/` empty
  throughout). This campaign's leg JSONs do not record tree state; the sibling
  GSCO campaign's per-leg `git_dirty_files` records corroborate a tree that was
  clean except this receipt through 08:45:13 EDT. The sibling campaign's doc
  files (`docs/jax_example_device_assignment.md`,
  `docs/receipts/wireframe_gsco_siblings_native_default.md`) are bracketed by
  the sibling's leg records to appearing between 08:45 and 08:58 EDT — possibly
  during this campaign's final legs (last at 08:48 EDT) — but both are `docs/`
  files outside anything this campaign executes, and are not this campaign's.
- **Box:** AMD Ryzen Threadripper 9970X (64 threads) + NVIDIA GeForce RTX 5090
  (32 GB, driver 595.84, CUDA 13.2)
- **Runtime:** JAX 0.10.0, Python 3.11.15, `JAX_ENABLE_X64=1`, fp64 throughout
- **Predecessor:** `docs/receipts/lm_qr_optin_route.md` (CPU verdict + Addendum A).
  This receipt does **not** supersede it; it adds the GPU axis and the width
  sweep that Addendum A's §A.4 explicitly said it did not have
  ("one wide point, not a sweep").
- **Artifacts:** `~/simsopt-campaigns/lm-qr-gpu-20260816/` — per-leg JSON under
  `legs/` (first sweep), `legs_clean/` (methodology-corrected sweep),
  `legs_reroute/` (§2c counterfactual, 42 legs) and `legs_stability/`
  (§2d distribution probe, 11 legs at 25 repeats); the kernel probe
  `qr_kernel_{cpu,gpu}.json`; `counterfactual_summary.json`;
  `fixture_conditioning.json`; Phase-1 captures; harness scripts; and the
  box-state record embedded in every leg file.

---

## 0. The question

`docs/receipts/lm_qr_optin_route.md` Addendum A established, on CPU, that dense
column-pivoted QR of the augmented system beats matrix-free GMRES by 4.6x at
400 x 60 — on **one** wide point. This campaign asks whether that makes the
LM_QR lane the repo's **second JAX-GPU-beats-CPU win**. That requires three
things the CPU receipt could not supply:

1. the lane must **run correctly on CUDA at all** (it never had been),
2. the single wide point must become a **width curve**, and
3. GPU-QR must beat **the best CPU configuration of either lane** — not merely
   beat GPU-GMRES, and not merely beat CPU-QR.

Bar (3) is the one that matters and is the one used throughout below.

---

## 1. Phase 1 — correctness on CUDA (untimed)

Harness: `phase1_correctness.py`. The three fixtures of the CPU receipt were
reconstructed from its §6 methodology (`_linear_fixture`, `_nonlinear_fixture`
from `tests/jax/solve/test_lm_qr_optin_route.py`, plus a 400 x 60 linear case
built the same way with `numpy` `default_rng` seed `20260816`).

**Reconstruction is proven exact, not assumed.** Run on CPU, the rebuilt
fixtures reproduce Addendum A.2's published digests bit for bit:

| record | `nit` | `fun` (hex float64) | SHA-256(`x`) prefix | matches A.2 |
|---|---|---|---|---|
| `linear_ls_40x8.qr` | 2 | `0x1.ffc7acfa3177dp+2` | `c49f7cf682c11db7` | yes |
| `linear_ls_40x8.gmres` | 2 | `0x1.ffc7acfa3177cp+2` | `5eb3d8a1f96a5ec2` | yes |
| `exp_fit_60x3.qr` | 17 | `0x1.f2f9787b58b80p-61` | `995aa8df1f93ee95` | yes |
| `linear_ls_400x60.qr` | 2 | `0x1.5dc7ed776f9dap+7` | `4337f4e4e4a9a426` | yes |

So the GPU comparison below is against the *same* problems the CPU receipt
measured, not lookalikes.

### 1.1 Result — the lane runs correctly on CUDA

Env: `SIMSOPT_BACKEND_MODE=jax_gpu_fast SIMSOPT_BACKEND_STRICT=1
SIMSOPT_PRECISION=fp64 JAX_PLATFORMS=cuda JAX_ENABLE_X64=1
XLA_PYTHON_CLIENT_PREALLOCATE=false MPI4PY_RC_INITIALIZE=false`.

All six (3 fixtures x 2 lanes) solves converge on `cuda:0`, to the same optima
as CPU, with **identical iteration counts**:

| fixture | lane | nit CPU/GPU | max abs GPU-vs-CPU diff in `x` | dev from reference |
|---|---|---|---|---|
| `linear_ls_40x8` | qr | 2 / 2 | 2.78e-17 | 2.657e-10 both |
| `linear_ls_40x8` | gmres | 2 / 2 | 4.16e-17 | 2.657e-10 both |
| `exp_fit_60x3` | qr | 17 / 17 | 2.22e-16 | 8.700e-10 both |
| `exp_fit_60x3` | gmres | 18 / 18 | 4.44e-16 | 3.326e-12 both |
| `linear_ls_400x60` | qr | 2 / 2 | 4.16e-17 | 1.294e-12 both |
| `linear_ls_400x60` | gmres | 2 / 2 | 1.08e-15 | 1.294e-12 both |

Every GPU-vs-CPU difference is at fp64 roundoff. Not bit-identical across
platforms, which is expected and not claimed.

**Pins that still hold on GPU:**

- the **not-byte-equal** QR-vs-GMRES pin holds on all three fixtures (§4 of the
  CPU receipt), and the `atol=1e-7` tolerance-equivalence pin holds alongside it;
- the dense-materialization cap still **fails closed on GPU**, with a
  byte-identical refusal message to CPU:
  `Levenberg-Marquardt dense QR solve requires residual Jacobian/Hessian
  artifacts totaling 3072 bytes in dtype float64, exceeding
  max_dense_linearization_bytes=1.`
- route strings are unchanged (`simsopt_lm_qr` / `lm-minpack-ondevice`).

### 1.2 Callback delivery on GPU needs `JAX_PLATFORMS=cuda,cpu`

Under `JAX_PLATFORMS=cuda` alone, **both** LM lanes fail any callback-attached
solve:

```
RuntimeError: jax.debug.callback failed to find a local CPU device to place the
inputs on. Make sure "cpu" is listed in --jax_platforms or the JAX_PLATFORMS
environment variable.
```

This is a JAX requirement on the `jax.debug.callback` staging path, not a
lane defect — it is symmetric across `lm_qr` and `lm_gmres`. With
`JAX_PLATFORMS=cuda,cpu` the computation still places on `cuda:0` and callbacks
deliver exactly as on CPU:

| lane | events CPU / GPU | iteration indices CPU / GPU | event type |
|---|---|---|---|
| `lm_qr` | 9 / 9 | `[6,10,11,12,13,14,15,16,17]` identical | `SimsoptLMQRCallbackEvent` |
| `lm_gmres` | 10 / 10 | `[6,10,…,18]` identical | `SimsoptLMGMRESCallbackEvent` |

(9 and 10 match Addendum A.7's per-solve counts.) **Operational note for any GPU
campaign using these lanes with instrumentation: set `JAX_PLATFORMS=cuda,cpu`.**

### 1.3 DEFECT (QR-lane-specific): callbacks trip `transfer_guard=disallow`

Under `SIMSOPT_JAX_TRANSFER_GUARD=disallow JAX_TRANSFER_GUARD=disallow` on GPU,
with problem data passed as runtime `residual_args` (so no closure constants
confound the test — see §1.4):

| case | guard=allow | guard=disallow |
|---|---|---|
| `lm_gmres`, no callback | OK | **OK** |
| `lm_gmres`, callback | OK | **OK** |
| `lm_qr`, no callback | OK | **OK** |
| `lm_qr`, callback | OK | **FAILS** |

Failure:

```
INVALID_ARGUMENT: Disallowed host-to-device transfer:
aval=ShapedArray(float64[8]), dst_sharding=SingleDeviceSharding(device=CpuDevice(id=0))
```

(Error text transcribed from the live session; the retained artifact
`phase1_guard_probe.json` truncates the surfaced form to
`INTERNAL: CpuCallback error calling callback: ...` at 300 characters.)

**Root cause.** `optimizer.py:4136-4140` registers the QR lane's step callback
as a *host-side adapter that runs JAX operations inside the host callback*:

```python
callback_token = _register_traceable_callback(
    None
    if callback is None
    else lambda flat_x: callback(_hostify_optimizer_tree(unravel(flat_x)))
)
```

`unravel` is the closure returned by `ravel_pytree`. It is **not** a host
function: `jax.flatten_util`'s `_unravel_list_single_dtype` calls
`lax.split(arr, sizes)`, a JAX primitive. So each callback event, running on the
host inside `jax.debug.callback`, stages its host array back onto a device to
perform the split. Under a `disallow` guard that is refused; without a guard it
is a silent per-event host->device->host round trip on GPU.

The default GMRES lane does not do this — it registers the user callback
directly (`optimizer.py:3652`), because its runner already delivers the iterate
in the caller's structure. Both lanes invoke the callback identically *in-trace*
(`optimizer.py:2623-2630` and `:4025-4032`); the asymmetry is entirely in what
is registered.

Note the trip fires even when `x0` is a **flat vector**, where Addendum A.1
describes the in-trace ravel as "a no-op reshape/concatenate of one operand" —
the host-side *un*ravel is a separate code path and is not a no-op there.

**Proposed fix (not applied — this campaign owns no `src/` edits).** Restore the
pytree on the host with `numpy`, so no JAX primitive executes inside the
callback. The structure is already known at registration time, so nothing new
must be plumbed:

```python
# at registration, from the same pytree ravel_pytree already consumed
leaves, treedef = jax.tree.flatten(x)
shapes  = [np.shape(leaf) for leaf in leaves]
dtypes  = [np.asarray(leaf).dtype for leaf in leaves]
splits  = np.cumsum([int(np.prod(s)) for s in shapes])[:-1]

def _host_unravel(flat_host):
    parts = np.split(np.asarray(flat_host), splits)
    return jax.tree_util.tree_unflatten(
        treedef,
        [p.reshape(s).astype(d) for p, s, d in zip(parts, shapes, dtypes)],
    )
```

then register `lambda flat_x: callback(_host_unravel(flat_x))`. This is strictly
less work than the current path (it removes a device round trip per event) and
`_hostify_optimizer_tree` becomes redundant on that path since the leaves are
already host arrays.

**Regression pin to add with the fix:** a callback-attached solve on both lanes
executed inside `jax.transfer_guard("disallow")`, asserting delivery. It fails
on today's code for `lm_qr` and passes for `lm_gmres`, so it discriminates.

**Severity:** low for correctness (numerics are untouched; the guard is opt-in),
moderate for GPU strictness campaigns — any receipt that runs the QR lane under
a strict transfer guard *with instrumentation* cannot currently be produced.

### 1.4 Two adjacent findings, reported not fixed

**(a) Closure-captured device arrays trip the guard in both lanes.** A residual
that closes over a concrete device array (the natural fixture spelling) makes
JAX hoist that array into an MLIR constant at lowering, which reads it back to
host:

```
INVALID_ARGUMENT: Disallowed device-to-host transfer: shape=(40,8), dtype=F64, device=cuda:0
  .../jax/_src/array.py:1108 in _array_mlir_constant_handler
```

This fires identically on `lm_qr` (`optimizer.py:4144`) and `lm_gmres`
(`optimizer.py:3657`), before any solver code runs. It is JAX constant-hoisting
behaviour, **not** a repo defect, and it is avoided by passing data through
`residual_args` — the path Addendum A.1 promoted `args` onto. Worth knowing
because it makes a naive strict-guard test look like a lane failure when it is
not; §1.3's probe controls for it explicitly.

**(b) Structured (dict) `x0` + callback fails in both lanes, on every platform.**

```
File ".../simsopt_jax/solve/dispatch.py", line 408, in legacy_callback
    x_host = host_array_after_ready(x, dtype=float)
TypeError: float() argument must be a string or a real number, not 'dict'
```

(Transcribed from the session; not retained verbatim in the campaign artifacts.
The cited `dispatch.py:408` line is verbatim in the tree.)

`dispatch.least_squares`'s callback adapter assumes a flat array iterate.
Reproduced on **CPU** as well as GPU, on both `lm_qr` and `lm_gmres`, so it is
pre-existing and platform-independent — outside this campaign's ownership, but
it means "callback + pytree decision vector" is currently unsupported through
`dispatch.least_squares` regardless of lane. Addendum A.2's structured-`x0`
battery record passes because it does not attach a callback to that case.

**Phase 1 verdict: the LM_QR lane is correct on CUDA.** It converges to the CPU
optima at fp64 roundoff with identical iteration counts, its pins hold, its cap
fails closed, and its callbacks deliver — subject to `JAX_PLATFORMS=cuda,cpu`
and to the §1.3 defect under a strict transfer guard.

---

## 2. Phase 2 — width ladder (timed)

Harness `phase2_width_leg.py`, driver `run_sweep_clean.sh`, artifacts in
`legs_clean/`. Protocol inherited from the CPU receipt's §6/A.3 so the numbers
are comparable to it: `maxiter=400`, one cold call, one warm-up call, then the
**median of 5** timed warm repeats through `simsopt_jax.solve.dispatch.least_squares`
(which returns host `numpy` arrays, so every timed call includes the
device->host sync — the wall is a completed solve, not an async dispatch).

Fixture at every rung: `A = default_rng(20260816).standard_normal((rows, cols))`,
`b = ...standard_normal(rows)`, `x0 = 0`. All rungs converge in `nit = 2`,
`nfev = njev = 3`, on every platform and every configuration, so the ladder is a
**matched-work** comparison throughout.

Two methodology corrections were applied before this sweep, both recorded in
`run_sweep_clean.sh`: the first pass (`legs/`) interleaved CPU and GPU legs, so
each GPU leg started while a 64-thread CPU leg was still draining; and the
1-minute loadavg was too slow a gate, so an explicit 2 s `/proc/stat` host-idle
delta gate was added before every timed leg.

### 2.1 Warm median (ms), median of 5, `maxiter=400`, end-to-end via dispatch

| rung | CPU qr | CPU gmres | CPU gmres-matfree | GPU qr | GPU gmres | GPU gmres-matfree |
|---|---|---|---|---|---|---|
| 400x60 | 3.42 | 11.96 | 11.32 | 3.12 | 267.43 | 264.48 |
| 800x120 | 16.06 | 40.70 | 39.88 | 10.13 | 328.32 | 329.05 |
| 1200x180 | 135.83 | 100.87 | 101.42 | 20.03 | 352.03 | 351.42 |
| 2000x300 | 902.24 | 164.07 | 185.78 | 639.81 | 482.15 | 415.03 |
| 3000x450 | 2250.99 | 351.65 | 314.46 | 1831.34 | 467.06 | 462.12 |
| 4000x600 | 3455.07 | 627.91 | 619.71 | 3034.04 | 539.00 | 527.15 |

### 2.2 Bar (3) for the lane **as shipped**

| rung | best CPU (any lane) | GPU qr | GPU-qr vs best-CPU | GPU-qr vs CPU-qr | GPU-qr vs GPU-gmres |
|---|---|---|---|---|---|
| 400x60 | 3.42 (qr) | 3.12 | **1.10x** | 1.10x | 85.7x |
| 800x120 | 16.06 (qr) | 10.13 | **1.59x** | 1.59x | 32.4x |
| 1200x180 | 100.87 (gmres) | 20.03 | **5.04x** | 6.78x | 17.6x |
| 2000x300 | 164.07 (gmres) | 639.81 | **0.26x** | 1.41x | 0.8x |
| 3000x450 | 314.46 (gmres_matfree) | 1831.34 | **0.17x** | 1.23x | 0.3x |
| 4000x600 | 619.71 (gmres_matfree) | 3034.04 | **0.20x** | 1.14x | 0.2x |

**As shipped, the LM_QR lane on GPU meets bar (3) only up to 1200 x 180, peaking
at 5.04x, and then collapses to 0.17-0.26x.** The GMRES lane on GPU never comes
close on this ladder (it is 17-86x behind GPU-QR below 2000 x 300 and only
overtakes it because GPU-QR collapses, not because it improves).

The collapse between 1200 x 180 and 2000 x 300 — a 1.7x width increase buying a
32x slowdown — is not an algebra shape. §2b identifies what it is, and §2d shows
that the wide GPU-QR entries in the table above are **not stable numbers**.

---

## 2b. Phase 2b — which kernel is the cost (pivoted vs unpivoted vs normal equations)

Harness `phase2b_qr_kernel_probe.py`, artifacts `qr_kernel_cpu.json` /
`qr_kernel_gpu.json` (written 2026-08-16 08:14 EDT). It times three
factorizations of the **same** augmented system `[J; sqrt(mu) I]` that
`_qr_lm_step` builds (`optimizer.py:3750`), at each ladder rung, median of 5,
under the same quiet gate:

- `pivoted_qr` — `jax.scipy.linalg.qr(..., pivoting=True, mode="economic")` plus
  triangular solve plus the permutation scatter. **This is what the lane ships**
  (`optimizer.py:3764-3774`).
- `unpivoted_qr` — the same with `pivoting=False` and no scatter.
- `normal_equations` — `solve(AᵀA, Aᵀb)` on the augmented system.

### 2b.1 The table (both platforms)

| rung | augmented shape | CPU pivoted | CPU unpivoted | CPU normal-eq | CPU pivot tax | GPU pivoted | GPU unpivoted | GPU normal-eq | **GPU pivot tax** |
|---|---|---|---|---|---|---|---|---|---|
| 400x60 | 460 x 60 | 0.941 | 0.732 | 0.132 | 1.29x | 0.867 | 0.627 | 0.184 | **1.38x** |
| 800x120 | 920 x 120 | 7.177 | 5.446 | 0.267 | 1.32x | 4.346 | 1.378 | 0.332 | **3.15x** |
| 1200x180 | 1380 x 180 | 106.004 | 73.497 | 0.852 | 1.44x | 8.293 | 2.895 | 0.601 | **2.87x** |
| 2000x300 | 2300 x 300 | 487.927 | 172.969 | 1.999 | 2.82x | 327.994 | 11.449 | 1.403 | **28.65x** |
| 3000x450 | 3450 x 450 | 1164.051 | 450.172 | 5.240 | 2.59x | 791.816 | 21.497 | 2.895 | **36.83x** |
| 4000x600 | 4600 x 600 | 1591.085 | 712.568 | 13.321 | 2.23x | 1463.671 | 37.025 | 5.395 | **39.53x** |

(All values ms, median of 5.)

### 2b.2 The mechanism statement

**Column-pivoted QR is the GPU cost.** The GPU pivot tax grows 1.38x at
400 x 60 to **39.53x at 4000 x 600**, while unpivoted QR on the same matrices
scales cleanly (0.63 -> 37.0 ms across a 10x width increase, i.e. essentially
the arithmetic). Normal equations are faster still (0.18 -> 5.4 ms). On CPU the
same pivot tax stays between 1.3x and 2.8x — pivoting is mildly expensive
everywhere and catastrophically expensive on the GPU specifically.

Column pivoting is a fundamentally more sequential algorithm than `geqrf`: it
selects a pivot column per step from a running column-norm scan, so it cannot be
blocked as aggressively, and the GPU's advantage — wide parallel work per
launch — is exactly what it cannot use. The step change is sharp: between 180
and 300 columns GPU pivoted-QR goes 8.3 -> 328 ms, a 40x jump for a 1.67x
column increase, which is a regime/lowering change rather than a scaling curve.
**This receipt does not identify the specific lowering or library path
responsible; it establishes only that the cost tracks `pivoting=True`.**

### 2b.3 The lane is ~2x the pivoted kernel — the attribution closes

Every rung converges in `nit = 2`, i.e. two LM steps, i.e. **two** calls to
`_qr_lm_step`. Comparing §2.1's lane wall against twice §2b.1's pivoted kernel:

| rung | plat | lane qr (ms) | 2 x pivoted kernel (ms) | lane / (2 x kernel) |
|---|---|---|---|---|
| 400x60 | cpu | 3.42 | 1.88 | 1.82x |
| 400x60 | gpu | 3.12 | 1.73 | 1.80x |
| 800x120 | cpu | 16.06 | 14.35 | 1.12x |
| 800x120 | gpu | 10.13 | 8.69 | 1.17x |
| 1200x180 | cpu | 135.83 | 212.01 | 0.64x |
| 1200x180 | gpu | 20.03 | 16.59 | 1.21x |
| 2000x300 | cpu | 902.24 | 975.85 | 0.92x |
| 2000x300 | gpu | 639.81 | 655.99 | 0.98x |
| 3000x450 | cpu | 2250.99 | 2328.10 | 0.97x |
| 3000x450 | gpu | 1831.34 | 1583.63 | 1.16x |
| 4000x600 | cpu | 3455.07 | 3182.17 | 1.09x |
| 4000x600 | gpu | 3034.04 | 2927.34 | 1.04x |

From 800 x 120 up the lane is within ~20% of two pivoted factorizations on both
platforms (the 1200x180 CPU cell is below 1.0x because both inputs to that ratio
are individually unstable — see §2d). At 400 x 60 the residual 1.8x is the
per-call host overhead Addendum A.4 already isolated (the `jax.eval_shape` cap
probe, 0.20-0.39 ms). **So the lane's wide-rung wall is the pivoted
factorization and essentially nothing else** — which makes the counterfactual in
§2c the decisive experiment rather than a curiosity.

---

## 2c. Phase 2c — the counterfactual: the real lane routed through unpivoted QR

Harness `phase2c_reroute_leg.py`, driver `run_reroute.sh`, artifacts
`legs_reroute/` (42 legs). **This is a measurement of the real lane, not a
projection.**

### 2c.1 Method, and why it is the lane and not a stand-in

`_qr_lm_iteration` (`optimizer.py:3790`) resolves `_qr_lm_step` as a **module
global at call time**, so replacing `simsopt_jax.geo.optimizers.optimizer._qr_lm_step`
before the first trace puts the replacement into the staged runner. The harness
does that **in its own process only**; no repo file is touched (`git status`
shows one untracked file, this receipt). Everything else — `dispatch.least_squares`,
`Driver.SIMSOPT_LM_QR`, `SimsoptLMQROptions`, the cap, the runner cache, the
`nit=2` trajectory — is the shipped path.

Three arms, one process each, one kernel each:

- **`pivoted`** — the shipped body re-expressed in the harness. This is the
  *control arm*: it exists so that pivoted and rerouted digests come out of
  identical harness code, in the same box-state window.
- **`unpivoted`** — identical except `pivoting=False`, no permutation scatter.
- **`normal`** — `solve(AᵀA, Aᵀb)` on the augmented system, carrying the
  conditioning caveat (§2c.5).

**The patch is proven live, not assumed.** Each replacement increments a
trace counter; a leg whose counter is zero raises and is recorded as a failure
rather than reported as a rerouted timing. Across `legs_reroute/` (42) and
`legs_stability/` (11): **all 45 `qr` legs report
`kernel_trace_count_after_cold = 1` and `patch.installed = true`; all 8 `gmres`
control legs report `0` and `false`**, as they must. Zero legs failed.

**Two independent checks that the control arm is the shipped lane:**

1. *Numerically.* The CPU `pivoted` arm at 400 x 60 reproduces Addendum A.2's
   published record **bit for bit** — `fun = 0x1.5dc7ed776f9dap+7`,
   `sha256(x) = 4337f4e4e4a9a426…` — and 400 x 60 is the identical fixture
   (`linear_ls_400x60`, same seed, same construction).
2. *Temporally.* The CPU GMRES control reproduces the `legs_clean` sweep across
   all six rungs to within 0.99x-1.15x, so the two sweeps' windows are
   comparable.

### 2c.2 Rerouted lane, warm median (ms), median of 5

| rung | GPU pivoted (shipped) | GPU unpivoted | GPU normal-eq | CPU pivoted (shipped) | CPU unpivoted | CPU normal-eq |
|---|---|---|---|---|---|---|
| 400x60 | 4.61 | 2.78 | 2.36 | 3.34 | 2.81 | 1.16 |
| 800x120 | 9.32 | 4.97 | 2.68 | 17.68 | 15.70 | 4.32 |
| 1200x180 | 20.57 | 10.35 | 4.97 | 166.81 | 320.10 | 8.43 |
| 2000x300 | 668.09 | 31.73 | 13.20 | 1460.99 | 939.13 | 22.54 |
| 3000x450 | 99.73 | 59.55 | 21.77 | 6776.68 | 3420.76 | 75.35 |
| 4000x600 | 231.44 | 102.08 | 40.30 | 2639.06 | 1276.55 | 116.25 |

The `pivoted` cells in this table disagree with §2.1's GPU-qr column by up to
~18x at the two widest rungs (99.73 vs 1831.34 = 18.4x; 231.44 vs 3034.04 =
13.1x). That is not a
harness discrepancy — it is the instability §2d characterises. **Every pivoted
number in this receipt, in either sweep, is a draw from a ~20x-wide
distribution.** The unpivoted and normal-equations cells are stable to a few
percent.

### 2c.3 The bar

Best available statistic per cell: 25 repeats where §2d measured them, 5
otherwise. "Shipped-lane bar" = the best CPU number available from any
configuration the repo actually ships (`lm_qr` as written, `lm_gmres` dense,
`lm_gmres` matrix-free), across both sweeps. "Any-kernel bar" additionally
allows the CPU lane the *same* one-line freedom the GPU lane is being given.

| rung | GPU qr/unpivoted (ms) | n | best CPU, shipped lanes | which | **vs bar (3)** | best CPU, any kernel | which | vs any-kernel bar |
|---|---|---|---|---|---|---|---|---|
| 400x60 | 2.78 | 5 | 3.34 | cpu qr (pivoted) | **1.20x** | 1.16 | cpu qr/normal | 0.42x |
| 800x120 | 4.97 | 5 | 16.06 | cpu qr (pivoted) | **3.23x** | 4.32 | cpu qr/normal | 0.87x |
| 1200x180 | 10.35 | 5 | 100.87 | cpu gmres | **9.74x** | 8.43 | cpu qr/normal | 0.81x |
| 2000x300 | 31.73 | 5 | 164.07 | cpu gmres | **5.17x** | 22.54 | cpu qr/normal | 0.71x |
| 3000x450 | 59.18 | 25 | 314.46 | cpu gmres-matfree | **5.31x** | 71.11 | cpu qr/normal | 1.20x |
| 4000x600 | 101.98 | 25 | 619.71 | cpu gmres-matfree | **6.08x** | 119.05 | cpu qr/normal | 1.17x |

**Result on bar (3) as this receipt's §0 states it: MET at every rung
measured**, 1.20x at 400 x 60 rising to 9.74x at 1200 x 180 and settling at
5.3-6.1x across 3000 x 450 - 4000 x 600. At the two widest rungs the GPU side is
stable at 25 repeats (unpivoted max/min = 1.0x); the bar cells themselves
(`gmres-matfree`) are n=5 and were not re-measured at 25, but the adjacent dense
CPU GMRES cells were (max/min 1.2-1.3x at 25), corroborating that side's
stability.

**Result on the harder question the bar was a proxy for: not established.** If
changing one kernel argument is on the table for the GPU lane, it is on the
table for the CPU lane too — and CPU normal-equations beats the GPU
unpivoted-routed lane at four of six rungs (0.42x-0.87x) and loses only
1.17-1.20x at the two widest. The width ladder therefore says the LM_QR lane's
wide-rung cost is **about the pivoting choice, not about the device**. A GPU
win exists, but it is ~6x over the best *shipped* CPU configuration and only
~1.2x over the best *reachable* CPU configuration, at the widest points measured.

### 2c.4 Solution agreement

Every arm at every rung converges in **`nit = 2`, `nfev = 3`, `njev = 3`** —
identical to the shipped lane. The algorithm differs, so bitwise agreement is
neither expected nor claimed; the measured deviation from the pivoted control is
at fp64 roundoff:

| rung | platform | kernel | `fun` (hex float64) | sha256(x)[:16] | max abs dev vs pivoted | **max rel dev vs pivoted** |
|---|---|---|---|---|---|---|
| 400x60 | gpu | pivoted | `0x1.5dc7ed776f9dap+7` | `8e7c11979ffd4a76` | — | — |
| 400x60 | gpu | unpivoted | `0x1.5dc7ed776f9dap+7` | `1b8cb21e800a0d3d` | 6.939e-17 | 5.648e-16 |
| 400x60 | gpu | normal | `0x1.5dc7ed776f9dap+7` | `0edeea4df9430074` | 6.939e-17 | 5.648e-16 |
| 400x60 | cpu | pivoted | `0x1.5dc7ed776f9dap+7` | `4337f4e4e4a9a426` | — | — |
| 400x60 | cpu | unpivoted | `0x1.5dc7ed776f9dbp+7` | `a350a2d712d8b36c` | 8.327e-17 | 6.777e-16 |
| 400x60 | cpu | normal | `0x1.5dc7ed776f9dbp+7` | `8426793758873f2a` | 8.327e-17 | 6.777e-16 |
| 800x120 | gpu | unpivoted | `0x1.6450c75fd1830p+8` | `f0ce0ed61f66adf7` | 9.194e-17 | 1.025e-15 |
| 800x120 | cpu | unpivoted | `0x1.6450c75fd1831p+8` | `d2172c0c65e76e9e` | 1.093e-16 | 1.218e-15 |
| 1200x180 | gpu | unpivoted | `0x1.e4de9e34e0fb4p+8` | `29e726f7cdee96c2` | 8.327e-17 | 9.079e-16 |
| 1200x180 | cpu | unpivoted | `0x1.e4de9e34e0fb3p+8` | `e0b805925aca21d3` | 8.674e-17 | 9.457e-16 |
| 2000x300 | gpu | unpivoted | `0x1.b663d86d75117p+9` | `4391b9c52e7df795` | 9.021e-17 | 1.238e-15 |
| 2000x300 | cpu | unpivoted | `0x1.b663d86d7510cp+9` | `73e621bd0eda2c42` | 9.714e-17 | 1.333e-15 |
| 3000x450 | gpu | unpivoted | `0x1.3cbdac335da49p+10` | `7ed5f8ccc71cf466` | 9.368e-17 | 1.551e-15 |
| 3000x450 | cpu | unpivoted | `0x1.3cbdac335da51p+10` | `ecd1df44008b03aa` | 9.021e-17 | 1.493e-15 |
| 4000x600 | gpu | unpivoted | `0x1.b543fd3303ab4p+10` | `4ab3d084db050333` | 9.714e-17 | 1.841e-15 |
| 4000x600 | cpu | unpivoted | `0x1.b543fd3303ab9p+10` | `6052f0b2de74ebdb` | 9.454e-17 | 1.792e-15 |

(Full 36-row table including the normal-equations arms in
`reroute_table.txt` §D.)

**Max relative deviation over all rungs, both platforms, both rerouted arms:
2.367e-15** (CPU normal-eq at 4000 x 600) — about 10 ulp. The largest `fun`
disagreement between the pivoted control and the unpivoted arm is 2 ulp of the
hex float64 (CPU 1200 x 180); including the normal-equations arms it is 3 ulp
(CPU 3000 x 450, pivoted vs normal). Deviation from the `numpy.linalg.lstsq`
reference is indistinguishable across the three kernels — identical to two
significant figures at the four narrower rungs (≈1.29e-12 at 400 x 60), to two
significant figures at 3000 x 450 (8.889-8.944e-15), and to within 1.6 % at
4000 x 600 (5.634-5.725e-15, both platforms) — i.e. the ladder cannot tell the
three factorizations apart numerically. §2c.5 explains why that is a limitation, not a result.

Digests are reproducible across processes: the 25-repeat §2d legs return the
same `sha256(x)` as the 5-repeat §2c legs for every cell they share.

### 2c.5 What the ladder cannot say about accuracy

The fixtures are random Gaussian matrices. Measured (`fixture_conditioning.json`):

| rung | kappa(J) | kappa(augmented) | kappa(JᵀJ) |
|---|---|---|---|
| 400x60 | 2.18 | 2.18 | 4.8 |
| 4000x600 | 2.25 | 2.25 | 5.1 |

**kappa ~ 2.2 at every rung.** Column pivoting exists to reveal numerical rank
and to order the reflections when columns are nearly dependent; on a matrix with
kappa = 2.2 it has nothing to reveal, so dropping it is free *by construction*.
Squaring kappa = 2.2 to 5.1 is likewise harmless, which is why the
normal-equations arm looks as accurate as QR here.

That is not the repo's regime. The 2026-07 mixed-precision campaign measured
**kappa(J) ~ 1034** on the real objective, i.e. **kappa(JᵀJ) ~ 1.07e6** (a
program-record figure carried in the campaign handoff and memory; no in-repo
artifact pins it, and it was not re-measured here). The
conditioning caveat on normal equations is therefore live for the real problem
and merely invisible on this ladder; and whether unpivoted QR is *also* safe
there is **untested** — see §3.

---

## 2d. Phase 2d — the shipped GPU pivoted lane time is not a stable quantity

Prompted by a 989% spread in one §2c leg. Same harness at `--repeats 25`,
artifacts `legs_stability/` (11 legs).

| cell | n | min | p25 | median | p75 | max | **max/min** |
|---|---|---|---|---|---|---|---|
| gpu 4000x600 qr/**pivoted** | 25 | 159.1 | 195.0 | 636.1 | 2258.1 | 3197.1 | **20.1x** |
| gpu 4000x600 qr/unpivoted | 25 | 101.0 | 101.6 | 102.0 | 102.4 | 103.0 | 1.0x |
| gpu 4000x600 qr/normal | 25 | 38.0 | 38.5 | 38.9 | 39.1 | 40.0 | 1.1x |
| gpu 3000x450 qr/**pivoted** | 25 | 95.9 | 105.3 | 162.4 | 1114.5 | 1859.8 | **19.4x** |
| gpu 3000x450 qr/unpivoted | 25 | 58.7 | 58.9 | 59.2 | 59.5 | 60.4 | 1.0x |
| gpu 3000x450 qr/normal | 25 | 20.7 | 20.9 | 21.1 | 21.4 | 21.7 | 1.0x |
| cpu 3000x450 qr/**unpivoted** | 25 | 328.9 | 893.0 | 1032.8 | 1131.9 | 1309.3 | **4.0x** |
| cpu 3000x450 qr/normal | 25 | 64.6 | 69.7 | 71.1 | 74.3 | 75.8 | 1.2x |
| cpu 4000x600 qr/normal | 25 | 114.1 | 116.0 | 119.1 | 120.1 | 123.2 | 1.1x |
| cpu 3000x450 gmres | 25 | 312.9 | 335.8 | 346.5 | 356.3 | 386.4 | 1.2x |
| cpu 4000x600 gmres | 25 | 573.4 | 615.0 | 626.9 | 665.6 | 759.1 | 1.3x |

Sorted GPU pivoted samples at 4000 x 600 (ms):
`159, 161, 164, 172, 175, 185, 195, 221, 267, 268, 346, 603, 636, 646, 838, 876,
1420, 1627, 2258, 2451, 2507, 2541, 2608, 3116, 3197` — **multi-modal across
20x on identical input, in one process, under a passing host-idle gate** (see
the contention disclosure below: the device itself was partially shared during
this probe).

Consequences, stated plainly:

1. **Any median-of-5 of the shipped GPU LM_QR lane at >= 2000 x 300 is a
   lottery ticket.** §2.1's GPU-qr column happened to sample the slow mode;
   §2c.2's pivoted arm happened to sample the fast mode. Neither is wrong;
   neither is a number.
2. **The `legs/` -> `legs_clean/` methodology fix was aimed at the wrong
   target.** The first sweep's non-monotone GPU-QR curve was attributed to CPU
   legs draining into GPU legs; interleaving was certainly a real defect, but
   the slow mode also appears in the pre-08:34 windows when this campaign was
   the device's sole client (§2.1's `legs_clean` draws and the 989% §2c leg),
   so contention was at most part of it.
3. **The mechanism conclusion survives and is strengthened.** Even the *fastest*
   pivoted draw at 4000 x 600 (159 ms) is slower than the unpivoted lane's
   *slowest* (103 ms), and the isolated pivoted kernel (§2b, 1251-1484 ms) sits
   in the slow mode. Pivoting is the cost under every draw.
4. **The bar-(3) comparison in §2c.3 is unaffected**, because it compares the
   *unpivoted* GPU lane (max/min 1.0x at 25 repeats) against the *GMRES* CPU
   lane, whose dense form is stable at 25 repeats (max/min 1.2-1.3x; the
   matfree bar cells themselves are n=5, not re-measured at 25).
5. **CPU dense QR is unstable too**, though less so (4.0x range, and the
   5-repeat CPU 3000x450 unpivoted median of 3420.76 ms sits well outside the
   25-repeat range 328.9-1309.3 ms). CPU normal-equations and CPU GMRES are the
   only stable dense CPU cells. This is why §2c.3's bar is built on GMRES.

**Contention disclosure for this phase.** Every §2d leg started with a passing
host-idle gate and a device list this campaign's `boxstate.py` classified as
baseline-only — but that classifier substring-matches `"code"` against process
paths, so it cannot see the sibling GSCO campaign's `.venv-qn-gpu` python
processes (whose path contains `/home/jungdaesuh/code/`). The sibling campaign's
own box-state records prove the overlap: they capture **this probe's own
processes** holding 564-944 MiB on the device while the sibling's batched
JAX-GPU legs ran, from 08:43:56, inside this probe's window (08:43-08:45:48) —
so the two GPU cells above were measured on a **partially shared device**, not
an exclusively held one. Host movement was also larger than a quiet box: `gpu
4000x600 qr/pivoted` rose from 10.53 to 34.84 1-minute loadavg across its run
(+24.3), `cpu 3000x450 qr/unpivoted` from 21.2 to 48.6 (part of its 4.0x range
may be host contention), `cpu 3000x450 gmres` moved -3.5, and `gpu 3000x450
qr/pivoted` +8.5; the remaining legs moved less. All 11 legs finished by 08:48 EDT. Two observations keep the
multi-modality finding standing despite this: the slow mode was first recorded
in `legs_clean/` and §2c **before 08:34**, when this campaign was the device's
sole client (the sibling was blocked on the sentinel — though its native CPU
legs were loading the host, so this is a sole-device observation, not a
quiet-host one); and the
unpivoted/normal-eq cells probed in adjacent windows of the same shared period
are flat (max/min <= 1.1x). But the 25-sample **distribution** itself (the
p25/p75 split above) was measured under partial device sharing and host load,
and should be read with that caveat.

No mechanism for the bimodality is established here. It is not input-dependent
(identical matrix every repeat, one process, one compiled executable), and the
20x fast/slow ratio with a sharp 40x regime jump between 180 and 300 columns is
not an arithmetic signature — but naming the responsible lowering or library
path would need instrumentation this campaign did not run.

---

## 3. Phase 3 — Boozer-shaped residual: NOT RUN

`phase3_boozer_leg.py` exists in the campaign directory (it drives
`_boozer_penalty_residual_vector`, the residual the production
`BoozerSurfaceJAX` lm routing uses, through both lanes). **It was never
executed** — the campaign was stopped before Phase 3, and no output artifact
exists. Nothing in this receipt is a statement about the production residual.

This is the load-bearing gap. §2c.5 shows the ladder's fixtures are
kappa ~ 2.2, so they cannot discriminate pivoted from unpivoted from normal
equations *numerically*, and §2b.2 shows the GPU pivot tax only appears above
~180-300 columns. The production Boozer problems this repo actually optimizes
are both far worse conditioned (kappa(J) ~ 1034) and, at the sizes named in the
formulation contract, **narrower** than the rungs where the reroute wins. Until
Phase 3 runs, the width-ladder win is a statement about 300-600-column random
Gaussian least squares and nothing else.

---

## 4. Verdict

1. **Correctness on CUDA: established** (§1). One QR-lane-specific defect found
   (§1.3, callbacks under a strict transfer guard) with a proposed fix and a
   discriminating regression pin; two adjacent pre-existing findings reported.
2. **Width curve: delivered** (§2), and it refutes the shape Addendum A.4's
   single wide point suggested. The shipped GPU LM_QR lane meets bar (3) only to
   1200 x 180 and then collapses.
3. **Mechanism: column-pivoted QR** (§2b). GPU pivot tax 1.38x -> 39.53x across
   the ladder while unpivoted QR scales with the arithmetic; the lane wall is
   two pivoted factorizations and ~nothing else.
4. **Counterfactual: measured, not projected** (§2c). Routed through unpivoted
   QR *in the real lane*, GPU **meets bar (3) at every rung — 1.20x to 9.74x,
   5.3-6.1x at the two widest** — converging in identical `nit`/`nfev`/`njev`
   with max relative solution deviation **1.8e-15** vs the pivoted control
   (2.4e-15 if the normal-equations arm is included).
5. **But the follow-up is NOT clearly worth chartering on this evidence**, for
   three independent reasons:
   - the same one-line freedom on CPU (normal equations) beats the rerouted GPU
     lane at 4 of 6 rungs and loses only ~1.2x at the widest (§2c.3);
   - the ladder's kappa ~ 2.2 fixtures are structurally incapable of pricing
     what pivoting buys (§2c.5), while the real problem is kappa(J) ~ 1034;
   - the production residual was never measured (§3).

   The honest summary is: **this receipt closes the mechanism question and
   leaves the routing question open.** It does not authorize a `src/` change; it
   identifies the one experiment (Phase 3 on the production Boozer residual,
   with a rank-deficiency stress case) that would decide it.
6. **New methodology finding** (§2d): the shipped GPU pivoted-QR lane time is
   multi-modal across ~20x at wide rungs. Any future receipt timing that lane
   must publish a distribution, not a median of 5.

---

## 5. Scope — what a real reroute would cost, and why that is a separate campaign

Everything in §2c was produced by an **in-process monkeypatch inside a campaign
harness**. Making it real means editing `_qr_lm_step` in
`src/simsopt_jax/geo/optimizers/optimizer.py`, which is a different kind of act
with its own review surface:

- **The handoff's Decisions section already ruled on this once.** "LM_QR routing
  reverted — abandons the `lm` <-> `lm-ondevice` byte-equality oracle,
  invalidates SHA-bound receipts (strict-results JSONL + TDD receipt + progress
  report), forces uncapped dense J/H materialization. Follow-up requires its own
  receipt campaign. Do not relitigate without that campaign."
  (`.handoffs/jax-gpu-vs-native-program.md`, Decisions.) That ruling stands.
- **The receipt invalidation is now measured, not predicted.** §2c.1 shows the
  pivoted control arm reproducing Addendum A.2's `linear_ls_400x60.qr` record
  bit for bit (`0x1.5dc7ed776f9dap+7` / `4337f4e4e4a9a426`), and §2c.4 shows the
  unpivoted arm producing `0x1.5dc7ed776f9dbp+7` / `a350a2d712d8b36c` on the
  same fixture. **Rerouting the shipped lane changes a digest that a tracked
  receipt publishes**, so the change necessarily drags `lm_qr_optin_route.md`
  Addendum A.2 with it, plus every pin keyed to it.
- **Pinned contract suite.** `tests/jax/solve/test_lm_qr_optin_route.py` (21
  test functions, 25 collected — 14 pre-existing plus 11 new, per
  `lm_qr_optin_route.md`) pins optimum agreement, the explicit not-byte-equal QR-vs-GMRES
  relation, and the cap behaviour. The `atol=1e-7` tolerance pin would survive a
  2.4e-15 perturbation; the digest-bearing records would not.
- **It is an algorithm change, not a tuning knob.** Pivoting is what makes the
  factorization rank-revealing. The augmented matrix `[J; sqrt(mu) I]` is full
  rank whenever `mu > 0`, so unpivoted QR is well-posed in the LM inner solve —
  but "well-posed" is not "equivalent", and this campaign's fixtures
  (kappa ~ 2.2) cannot price the difference. Normal equations are a strictly
  larger change: kappa(JᵀJ) ~ 1.07e6 on the real problem.

**What this receipt establishes is only whether the follow-up is worth
chartering.** Its answer is: *not yet, and here is the experiment that would
settle it* — Phase 3 on the production Boozer residual, at production widths,
with an explicitly rank-deficient or near-deficient stress case, comparing all
three kernels on accuracy first and speed second, and publishing the pivoted
lane as a distribution per §2d.

---

## 6. Reproduce

Campaign directory `~/simsopt-campaigns/lm-qr-gpu-20260816/`, repo at
`829d92f23`, `.venv-qn-gpu/bin/python`, one file per process.

```bash
# Phase 1 (untimed correctness), Phase 2 (width ladder), Phase 2b (kernel probe)
./run_sweep_clean.sh                       # -> legs_clean/
python phase2b_qr_kernel_probe.py --label gpu --require-gpu-quiet \
    --out qr_kernel_gpu.json               # CPU leg: same, --label cpu

# Phase 2c (counterfactual reroute) and Phase 2d (distribution probe)
./run_reroute.sh                           # -> legs_reroute/  (42 legs)
./run_stability.sh                         # -> legs_stability/ (11 legs, n=25)

# Tables
python build_table.py                      # LEGDIR=legs_clean by default
python build_reroute_table.py              # -> reroute_table.txt
```

GPU legs: `SIMSOPT_BACKEND_MODE=jax_gpu_fast SIMSOPT_BACKEND_STRICT=1
SIMSOPT_PRECISION=fp64 JAX_PLATFORMS=cuda JAX_ENABLE_X64=1
XLA_PYTHON_CLIENT_PREALLOCATE=false MPI4PY_RC_INITIALIZE=false` plus the warm
persistent-cache trio. CPU legs: `CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu
JAX_ENABLE_X64=1 MPI4PY_RC_INITIALIZE=false`. Every leg gates on
`boxstate.py::wait_for_quiet` (no foreign compute app on the device, loadavg
< 32) plus a 2 s `/proc/stat` host-idle delta below 12%, and embeds the
resulting box state in its own JSON.

**No `src/` or `tests/` file was modified by any phase of this campaign.** The
§2c reroute is a process-local monkeypatch in the harness, asserted live via a
trace counter recorded in every leg (`result.kernel_trace_count_after_cold`).
