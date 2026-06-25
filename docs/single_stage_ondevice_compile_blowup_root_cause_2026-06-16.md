# Single-Stage `ondevice` Compile Blowup — Root-Cause Investigation (2026-06-16)

> Status: RESOLVED (architectural). Live A100 reproduction + 3-agent code trace + XLA-dump
> discriminator + `JAX_LOG_COMPILES` confirmation. Pole = host-side **construction** of the
> fused `_value_and_grad_for` objective (pre-XLA-compile). Exact internal call (medium
> confidence) is pinnable by a local CPU py-spy — see §6/§8.

## 0. The observation

On RunPod A100, running the single-stage init at the example's **default production
resolution**:

```
python examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py \
  --backend jax --mpol 8 --ntor 8 --nphi 255 --ntheta 64 \
  --optimizer-backend scipy-jax --init-only \
  --warm-start-run-dir <donor> --jax-runtime-seed-spec <spec>
```

- BFGS Boozer least-squares **compiled + ran on GPU, nvlink-clean** (`bfgs-ondevice solve
  - success=False iter=1500, ||grad||_inf=3.3e-6`).
- `jit_traceable_newton_polish_run_solver` **compiled + cached** ~15 s later.
- Then a **single XLA compilation pegged ~15 CPU cores for 50+ min**, GPU idle (431 MiB /
  0 %), host RSS **flat at 10.7 GB** (the pod has ~1.9 TB RAM — *no* memory blowup), no
  further cache entry written. Not a hang (CPU genuinely burning: `delta_8s≈12080` jiffies
  = ~15 cores), not nvlink, not OOM, not the Newton skip.

This is the **compile-*time*** dimension of the known ondevice blowup. The earlier 422 GiB
event was the **memory** dimension of the same monolithic-graph problem; here memory is
modest but compile time is unbounded.

## 1. Root cause (resolved)

**RC-1 — The per-eval objective is one monolithic fused `jax.jit`.**
Under `--optimizer-backend scipy-jax`, routing maps the inner Boozer solve →
`ondevice` (`src/simsopt_jax/geo/optimizers/single_stage_routing.py:74-91`), and
`_value_and_grad_for` (`src/simsopt_jax_adapters/geo/surface_objectives_traceable.py:1376`,
jit at `:1414`) **inlines the entire inner solve into one XLA module**: warm-start predict
+ BFGS `lax.while_loop` + dense-Hessian-LU **Newton polish** (`run_code_traceable`,
`src/simsopt_jax_adapters/geo/boozer_surface.py:5320`) + **implicit-diff adjoint LU solve**
(`_traceable_total_gradient` → `lu_solve`, `surface_objectives_traceable.py:851,448`).
`scipy-jax` host-drives only the **outer** L-BFGS-B; **every outer eval re-solves the
surface inside the JAX graph** (no host cache of the surface). The gradient is implicit
(not unrolled), so the backward graph embeds an adjoint linear solve on the converged PLU
factors — bounded, but dense at production shapes.

**RC-2 — The compile cost is shape-driven, not graph-size-driven.**
Lowering the inner kernel at mpol2/nphi31 vs mpol8/nphi255 (via `jax.jit(fn).lower(*a).as_text()`,
no compile) shows the **HLO op-count is invariant (~2 %)** — identical `dot_general` /
`lu_factor` / `triangular_solve` ops. What explodes is **XLA's codegen/optimization cost on
those dense-LA ops as operand shapes scale**: grid (nphi·ntheta) 496 → 16 320 (**33×**),
surface dofs (∝ mpol·ntor) 27 → 257 (**~9.5×**), dense Hessian n_dofs². Empirically the same
forward-objective graphs (`jit_f`, `jit__forward_result_for`, `jit__value_and_grad_for`)
compiled in **~2 min at mpol2/nphi31** (smoke run) vs **50+ min at mpol8/nphi255** — same
backend, same graph, only resolution changed.

**RC-3 — No un-fuse lever exists (the TORAX pattern is absent).**
There is no configuration that compiles the field residual+Jacobian as separate GPU kernels
and **host-drives the inner Boozer iteration** (the TORAX design: compile the PDE residual
& Jacobian, keep the iterative-solver control flow on the host). The two real options are:
- `ondevice` inner (the `scipy-jax` default) → fully fused monolith on GPU (this blowup);
- `--boozer-optimizer-backend scipy` → the **whole** inner solve incl. field LU on CPU/host
  (`boozer_surface.py:3373-3392`), i.e. trade compile-blowup for a slow CPU inner solve.
- `scipy-jax-fullgraph` does **not** help: the historical `io_callback` host-bridge was
  retired (`single_stage_banana_example.py:13860`); it only changes the outer decision-vector
  ordering, not the fused graph contents.

## 2. Which jit is the pole — discriminator (Agent A vs Agent B)

Two candidate poles were proposed by the parallel tracers:
- **Agent A:** `jit_reporting_metrics_from_solution(include_distance_metrics=True)` — a
  separate, **init-only, gratuitous** pairwise curve-curve/curve-surface/surface-vessel
  distance graph over nphi255×ntheta64 (`surface_objectives_traceable.py:2234`; gate
  `include_distance_metrics = not benchmark_mode`, `single_stage_banana_example.py:6868`).
  → cheap fix (skip in init / `--benchmark-mode`).
- **Agent B:** the fused `_value_and_grad_for` itself. → architectural; also a production
  cold-compile cost.

**Discriminator experiment (live, A100): `--benchmark-mode` + `--xla_dump_to`.**
- `--benchmark-mode` sets `include_distance_metrics=False`, removing Agent A's graph. The
  run **still wedged at 41 min** on the same CPU-pegged compile → **not** the distance graph.
- The XLA dump showed modules only up to `module_1395.jit_traceable_newton_polish_run_solver`
  (`.after_optimizations` at 04:58:49); **no `reporting_metrics` module was ever reached**,
  and **no module after the Newton was emitted for 35 min** while 15 cores stayed pegged.
  Execution is wedged at the **`_value_and_grad_for` compile**, which *precedes* reporting.

⇒ **Agent B confirmed.** The 50-min pole is the fused per-eval value-and-gradient objective,
not the gratuitous init-only distance graph.

**`JAX_LOG_COMPILES=1` (warm cache) pinpoints the phase.** The run logs, in order:
`bfgs-ondevice solve` (ran) → `Compiling jit(traceable_newton_polish_run_solver)` →
`Persistent compilation cache hit ...` → `Finished XLA compilation of jit(traceable_newton_polish_run_solver) in 1.29 sec`
— **then total silence for 28+ min** while 16 cores stay pegged and the GPU is idle. The next
function (`_value_and_grad_for`) **never prints `Finished tracing` or `Compiling jit(...)`**
(grep for `value_and_grad|forward_result_for|reporting_metrics` in the log = **0 lines**). In
JAX, `Finished tracing X` is emitted after Python-level jaxpr construction and *before*
lowering/XLA-compile. Its absence means the wedge is in the **host-side construction of the
fused value-and-gradient graph — before XLA compilation of it even begins** (consistent with
the XLA dump never receiving a `_value_and_grad_for` module). This **refines RC-2**: the cost
is not XLA's optimization/codegen passes but JAX-side construction (trace + implicit-diff
gradient transform, and the 16-core native burn points to repeated eager execution of compiled
dense-LA kernels — Jacobian materialization / `lu_factor` — during the adjoint setup, trip
count scaling with resolution). Pinning the exact call needs py-spy (§8).

## 3. Why `--init-only` pays this at all

`--init-only` does **not** early-exit; it runs the whole `__main__` body
(`single_stage_banana_example.py:13276`) with `skip_outer_optimizer=True` (`:14451`). After
surface generation it evaluates the **seed initial objective** via `_value_and_grad_for`
(`:14690`) — that is the pole. (The reporting snapshot at `:16331`, Agent A's candidate,
runs *after* and is reached only once value_and_grad finishes.) So even a single objective
eval pays the full monolithic cold compile.

## 4. Relationship to the committed Newton-skip fix (`79392d40b`)

**Independent.** The Newton-skip fix correctly makes the GPU **run** (not skip) the Boozer
Newton — proven here: `jit_traceable_newton_polish_run_solver` compiled + cached on the GPU
15 s after BFGS. The compile blowup is a **separate, pre-existing** property of the fused
objective graph and is **not** introduced or worsened by the fix. Production amortizes it via
a warm `JAX_COMPILATION_CACHE_DIR` (`project_scipy_jax_gpu_compile_bound`: once-slow, Case A);
the RunPod reproductions were artificially cold (and the dump flag busted the cache key).

## 5. Fix directions (ranked)

1. **Build the missing host-driven inner-Boozer path (TORAX pattern).** Keep the field /
   Biot-Savart residual+Jacobian as separately-compiled GPU kernels; host-drive the Boozer
   Newton/LS *iteration* in Python/numpy so it is **not** fused into the per-eval objective
   graph. The sole embed site is `run_code_traceable` (`boozer_surface.py:5320`). This shrinks
   each XLA module to a fixed, resolution-light kernel and removes the super-linear codegen.
2. **Warm-cache amortization (status quo for production).** One cold compile per (resolution,
   shape) signature, then cached. Acceptable for long runs; painful for iteration / per-seed
   sweeps that change shapes.
3. **`--boozer-optimizer-backend scipy`** — existing un-fuse, but moves the whole inner solve
   (incl. field LU) to CPU; slow inner solve, defeats GPU.
4. **Init-only cheapening (orthogonal):** `--benchmark-mode` / `include_distance_metrics=False`
   removes the *separate* distance-reporting compile (Agent A's graph). Does **not** fix the
   value_and_grad pole, but avoids paying a *second* large compile in init/reporting.

## 6. Evidence log

- Live A100 pod `m60dcnkic4ffoy`; cold runs at mpol8/nphi255.
- Thread states: 15 `R` / 1225 `S`; `delta_8s≈12080` jiffies ⇒ ~15 cores burning; RSS flat
  10.69 GB; GPU 0 % / 431 MiB throughout the wedge.
- XLA dump: highest module `1395` (Newton); none after for 35 min; no `value_and_grad` /
  `reporting_metrics` module emitted.
- mpol2/nphi31 smoke compiled `jit_f`/`jit__forward_result_for`/`jit__value_and_grad_for` in
  ~2 min (02:38→02:40 cache stamps).
- Op-count invariance: inner residual 42 ops at both resolutions; newton-step (Jac+JᵀJ+LU)
  245 ops at both; only operand shapes scale.
- `JAX_LOG_COMPILES=1` warm-cache run: BFGS compiled+ran, Newton **cache-hit** (1.29 s),
  then **28 min of log silence** with 16 cores pegged / GPU idle; `_value_and_grad_for` never
  reached even its `Finished tracing` line (grep = 0). ⇒ wedge is host-side **construction**
  of the fused objective, *before* its XLA compile starts.
- Pod `m60dcnkic4ffoy` deleted 2026-06-16 after evidence capture (billing stopped).

## 7. Key files

- `src/simsopt_jax/geo/optimizers/single_stage_routing.py:46-91` — backend routing table.
- `src/simsopt_jax_adapters/geo/surface_objectives_traceable.py:1322,1357,1376,1414,851,448,2234`
  — fused forward / value_and_grad / adjoint / reporting jits.
- `src/simsopt_jax_adapters/geo/boozer_surface.py:5320-5409` (`run_code_traceable`, ondevice
  embed), `:3373-3392` (scipy inner pulls field LU to host).
- `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:13276`
  (`__main__`), `:14451` (init-only), `:14690` (seed value_and_grad — the pole), `:6868`
  (`include_distance_metrics` gate), `:16331` (reporting snapshot).

## 8. Open: pin the exact construction call (local py-spy)

The exact internal call that burns 16 cores for ~50 min during `_value_and_grad_for`
construction is **medium confidence** (hypothesis: eager dense-Jacobian materialization /
`lu_factor` executed in a host loop over modes/dofs during the implicit-diff adjoint setup).
The A100 pod could **not** py-spy (container lacks `SYS_PTRACE`). The construction is host-side
and **platform-independent**, so it reproduces on local CPU JAX:

```
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 JAX_LOG_COMPILES=1 \
  python examples/.../single_stage_banana_example.py --backend jax \
  --mpol 8 --ntor 8 --nphi 255 --ntheta 64 --optimizer-backend scipy-jax --init-only ...
# when the log goes silent after the Newton compile, in another shell:
py-spy dump --pid <pid>        # or: py-spy record -o wedge.svg --pid <pid> --duration 60
```

The Python stack at the wedge names the exact function/loop. Fix then targets that call
(vectorize the construction, or host-drive the inner solve per RC-3) rather than the whole
graph.
