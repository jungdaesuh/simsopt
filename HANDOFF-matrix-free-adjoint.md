# HANDOFF — dense-linearization chunk fix → matrix-free adjoint (Gate 4) for the single-stage parity matrix

> Last updated: 2026-06-22 00:32 EDT · Status: **Chunk fix IMPLEMENTED + PROVEN + crucible-PASS, but UNCOMMITTED. Gate-4 matrix-free-adjoint plan WRITTEN.** Three open decisions for the user: (1) commit the chunk fix or hand to codex; (2) execute the matrix-free-adjoint A/B; (3) stop the A40. This continues AFTER the 4-seed parity matrix completed (see `HANDOFF-parity-matrix.md`, that task DONE).

## 1. Goal
Continuation arc after the 4-seed cpp↔jax-GPU single-stage parity matrix was completed + committed. The user wants to run the parity matrix / single-stage on GPU with **`XLA_PYTHON_CLIENT_PREALLOCATE=true`** (the JAX-recommended default). Under preallocation the warm-eval crashed with `RESOURCE_EXHAUSTED: Failed to load in-memory CUBIN: CUDA_ERROR_OUT_OF_MEMORY` in the inner Boozer re-solve's **dense final linearization**. User directive: **"no hacks, proper fixes only"** (rejected `preallocate=false` and `MEM_FRACTION` tuning as band-aids).
**Definition of done:** (a) a root-cause fix that lets the dense lane run under `preallocate=true` with bit-identical numerics + preserved parity; (b) a clear path (design + decision) to the true matrix-free-adjoint endgame.

## 2. Where we are right now
The **root-cause chunk fix is done, validated end-to-end, and crucible-PASS — but UNCOMMITTED** (it lives in `src/simsopt_jax/geo/optimizers/optimizer.py:3608`, commingled with concurrent codex edits to the same file). A **Gate-4 matrix-free-adjoint implementation plan** is written (`docs/matrix_free_adjoint_gate4_implementation_plan_2026-06-22.md`). The immediate situation: nothing of mine is committed since `223ddb37c`; the user must decide whether to commit the chunk, run the matrix-free A/B, and/or stop the (idle) A40.

## 3. NEXT ACTIONS (start here on resume)
1. [ ] **Decide + (if yes) commit the chunk fix.** It's `optimizer.py:3608` (`lax.map(..., eye, batch_size=8)` in `_materialize_dense_linear_operator`) + the rationale comment above it. `optimizer.py` is ` M` with codex's *other* in-flight edits too, so do NOT `git add` the whole file — use the scoped temp-index `commit-tree` recipe (see §6) committing ONLY the chunk hunk, OR hand the hunk to codex (codex owns concurrent `optimizer.py` work). Suggested msg: "fix(jax): chunk dense linear-operator assembly (batch_size=8) so preallocate=true doesn't OOM; mirrors dcd70a2ae".
2. [ ] **Execute the matrix-free-adjoint plan** `docs/matrix_free_adjoint_gate4_implementation_plan_2026-06-22.md` — start at its Step 1 (read-only wiring trace: which contract the parity matrix resolves to; locate the live byte-parity enforcement; confirm the existing `custom_vjp`+operator-GMRES adjoint is selectable for the exact-Jacobian lane). Then the flag + the A/B.
3. [ ] **Stop the A40** if no more GPU work (`runpodctl pod stop <id>`; $0.44/hr; currently idle 0 MiB).

## 4. Environment & how to run
- cwd / repo / branch: `/Users/suhjungdae/code/columbia/simsopt-pr-jax-port-clean` / `pr/jax-port-clean` / HEAD **`223ddb37c`**.
- ⚠️ Working tree: **71 files** dirty (concurrent codex) + **4 pre-staged not-mine files in the index** (lbfgs doc, `_lbfgs.py`, `_lbfgsb_scipy.py`, `test_boozersurface_jax_private.py`) — never `git add -A`; commit only your hunks via temp-index.
- **A40 pod (LIVE, idle):** `ssh -o StrictHostKeyChecking=no -i ~/.runpod/ssh/RunPod-Key-Go -p 22081 root@194.68.245.6` (full inline ssh, NOT a `$SSH` var; port may change on restart). Pod repo (file copy): `/workspace/simsopt_lbfgs_fullcpu_a40_20260619T2155Z`. Venv: `/workspace/venvs/simsopt-lbfgs-a40-py312/bin/python` (jax 0.10.0, CUDA12, ptxas 12.8). 46 GB GPU.
- **rsync local→pod (NOT git):** `rsync -rlt --no-owner --no-group --no-perms -e "ssh -o StrictHostKeyChecking=no -i ~/.runpod/ssh/RunPod-Key-Go -p 22081" <relpath> root@194.68.245.6:/workspace/simsopt_lbfgs_fullcpu_a40_20260619T2155Z/<relpath>`.
- **Matrix launcher (committed):** `bash benchmarks/run_parity_matrix_pod.sh cuda <out.json> jax-gpu --equilibria-dir /workspace/equilibria --seeds /workspace/matrix_seeds/april285` (drop `--no-warm-eval` for the warm test). Run detached: `nohup env PYTHONUNBUFFERED=1 bash …launcher… > /tmp/x.log 2>&1 & disown`.
- **⚠️ `preallocate=true` is the LAUNCHER DEFAULT here** — the matrix bypasses the repo's `xla_gpu_preallocate=False` default (the `repo_bootstrap` setdefault fires too late, after jax GPU init). PROVEN: `m_cached` (launcher default) cold=56s vs `m_warm4` (explicit `preallocate=false`) cold=106s. So to force `preallocate=false` you must set the env var explicitly pre-process.
- **Pod pkill gotcha:** `pkill -f single_stage_objective_parity_matrix` will kill YOUR ssh shell (its argv contains the pattern) → exit 255. Use a bracket regex: `pkill -9 -f '[s]ingle_stage_objective_parity_matrix'`.
- Local lint: `python3 -m py_compile / pyflakes / ruff check <file>`. Repo tests: `./.conda-env/bin/python` (py3.11); base py3.13 fails simsoptpp import. Pod tests use the venv.

## 5. Done so far (with evidence)
- [x] **Committed `db14e992e`** (me): native-mpol (iota,G)-recovery robust fix (eval.py, parity_matrix.py, smoke_fixture.py). The 4-seed matrix is machine-precision (see `HANDOFF-parity-matrix.md`).
- [x] **Committed `3c16a7888`** (codex): self-check hardening (constraint_weight SSOT, full-dof + all-free assert, (iota,G)-convergence + drift asserts) — captured my A/C/E doc/null-safety fixes too.
- [x] **Committed `223ddb37c`** (me): softened the overstated `_recover_consistent_iota_G` docstrings (eval.py).
- [x] **Chunk fix IMPLEMENTED + PROVEN (UNCOMMITTED)** — `optimizer.py:3608`: `_materialize_dense_linear_operator` `lax.map` now `batch_size=8` (shared by dense Hessian `:3645` + Jacobian `:3661`), mirroring `src/simsopt_jax_adapters/geo/boozer_surface.py:5769` (commit dcd70a2ae). VALIDATED on A40: (1) `test_materialize_dense_linear_operator_matches_linear_map` → **1 passed (90.8s)** bit-identical; (2) `preallocate=true` warm-eval (`/tmp/m_fix.json`) → **EXIT=0, NO OOM** (cleared the warm jax re-solve at 97% GPU / 34.9 GB 75%-pool — the prior crash point), **value_abs_diff=0.00e+00, grad_rel=1.47e-16**; jax cold 61s/warm 315s, cpp cold 42s/warm 236s. Lint clean.
- [x] **Crucible PASS** on the chunk (2 adversarial agents): no correctness defect (remainder/edge-cases verified vs JAX 0.10 source); fixed one in-scope nit (comment qualified: bit-exact for the linear Jacobian, ~1e-16 for the Hessian's reducing HVP). Recorded **mistake-book Pattern 81** (incomplete-sibling-fix).
- [x] **optax/torax study done** (2 agents) — see §9; confirmed direction + found the matrix-free machinery already exists here.
- [x] **Gate-4 plan WRITTEN** (UNTRACKED): `docs/matrix_free_adjoint_gate4_implementation_plan_2026-06-22.md`.
- [x] Memory updated: `project_parity_matrix_completion_2026_06_21` (matrix done); mistake-book Pattern 3 (occurrence #3) + Pattern 81.

## 6. Key decisions & rationale
- **Chunk fix (batch_size=8) is the proper root-cause fix for the dense lane**, not a hack — it's the missing half of dcd70a2ae's fix (DRY/SSOT), bit-identical, bounds peak memory so `preallocate=true` works. Literal `8` mirrors the sibling (user chose this minimal form over deriving K from `max_dense_jacobian_bytes`).
- **`preallocate=true` is the user's target** (JAX-recommended performance default), even though the repo *defaults* `xla_gpu_preallocate=False` for `jax_gpu_*` modes (`runtime.py`). Flipping the repo-wide default is a SEPARATE policy decision, NOT in scope.
- **Matrix-free adjoint is the endgame, chunk is the bridge.** The forward solve is ALREADY matrix-free (`optimizer.py:5135-5136`); only the FINAL dense linearization (the adjoint factor) is dense. The endgame = route THAT through the existing operator-GMRES/`custom_vjp` machinery — gated by the **byte-parity contract** (dense factor shared forward+adjoint). This is a design DECISION, not a mechanical swap.
- **Temp-index `commit-tree` for scoped commits** (interactive `git add -p` unavailable; pre-staged not-mine hunks present): `GIT_INDEX_FILE=/tmp/ti git read-tree HEAD` → `git add <only my file(s)>` → `write-tree` → verify `git diff --name-only HEAD $T` == exactly my files → `commit-tree -p HEAD` → `update-ref refs/heads/pr/jax-port-clean` → `git restore --staged <my files>` to resync. NEVER `git reset/stash/clean` or `git add -A`. (NOTE: for the chunk, `optimizer.py` ALSO has codex's hunks — a whole-file `git add` would capture those too; need `git add -p`-equivalent or hand to codex.)

## 7. Dead ends / do NOT retry
- **`XLA_PYTHON_CLIENT_PREALLOCATE=false`** — works (no OOM) but ~2× slower cold (on-demand alloc) AND the user explicitly rejected it as a band-aid. (It IS the repo's committed GPU default, but the user wants `true`.)
- **`XLA_PYTHON_CLIENT_MEM_FRACTION=0.5`** — tested (`m_mf50`, got past the OOM point), confirms the mechanism (scratch-outside-pool), but rejected as a hack. Don't ship it.
- **Treating the warm-eval OOM as a forward-solve problem** — it's the FINAL dense linearization (adjoint), not the forward (forward is matrix-free GMRES).
- **`git add optimizer.py` to commit the chunk** — would sweep in codex's concurrent edits to the same file. Scope the hunk only.
- **Concluding torax shows matrix-free implicit-diff** — torax's `custom_root` `tangent_solve` is DENSE (its root is ~100-dim); it shows the BOUNDARY, not a drop-in matrix-free solve.

## 8. Open questions / blockers (need user)
- **Commit the chunk fix, or hand to codex?** (It's proven; uncommitted; commingled with codex's optimizer.py edits.)
- **Run the matrix-free-adjoint A/B?** (The plan's core experiment; multi-hour GPU + a byte-parity decision.)
- **Byte-parity contract:** is forward/adjoint byte-exact factor-sharing still required by any consumer, or can the exact-Jacobian lane relax it to go matrix-free? (The gating decision for Gate 4.)
- **Stop the A40?**

## 9. Mental model (hard-won context)
- **Forward inner Newton = matrix-free already** (`optimizer.py:5135-5136`: JVP + GMRES, dense rebuilt only at the final iterate, gated by `materialize_dense_linearization`). The OOM/warm-cost is purely the FINAL dense linearization (the IFT adjoint factor).
- **The matrix-free machinery ALREADY EXISTS in this repo** (so the endgame is ROUTING, not building): operator GMRES `_run_operator_gmres` (`optimizer.py:4129`), `_gmres_solve_exact_newton_system` (`:4162`); `@jax.custom_vjp` IFT adjoints (`src/simsopt_jax/solve/minimize_runtime.py:62`, `src/simsopt_jax/core/_root.py:103` — this repo's `custom_root` analog, `boozer_residual.py:146,166`); a matrix-free contract default (`src/simsopt_jax/solve/optimistix/contracts.py:28` `materialize_dense_linearization=False`). The DENSE byte-parity lane is `src/simsopt_jax/solve/simsopt/contracts.py:65` (`=True`), threaded by `solve/dispatch.py:168,539`.
- **Why jax warm > cpp warm** (315 vs 236s): ~19s fixed cold gap (jax host-side fused value/grad construction; cpp has none) + ~60s warm-specific = the dense-linearization ASSEMBLY (jax = ~80 sequential GPU BiotSavart-JVP batches after chunking; cpp = native BLAS). NOT iteration count (jax did 4 solve iters vs cpp 205). Matrix-free adjoint would remove this. Caveat: ~25s pod load-variance.
- **GPU loses at SMALL problem size** (nphi127, ~600 dofs → overhead-bound) but WINS at scale: at nphi255 `corner_ms`, jax-GPU 102s < cpp 113s even cold. So the warm-slowness is a small-size + chunking-tradeoff artifact, not a GPU deficit.
- **`surface_objectives_traceable.py` EXISTS** at `src/simsopt_jax_adapters/geo/` (3762 lines) — a prior draft wrongly said it was "refactored away". The byte-parity (lu,piv) shared forward/adjoint factor IS `_traceable_solve_plu_linearization` (`surface_objectives_traceable.py:~431-446`, Phase-2 contract §5.3). The punch-list Gate-4 path was correct all along. (corrected 2026-06-22; the prior claim was a wrong-scope grep — searched `simsopt_jax/`, not `simsopt_jax_adapters/`.)
- **The warm-eval is a benchmark artifact** — the synthetic `x0+1e-8` perturbation forces a heavy 205-iter re-solve + full dense linearization that PRODUCTION avoids (it warm-starts each outer step from the prior converged surface). So `wall_warm_s` is not a production steady-state number.
- **Second un-chunked dense site (Pattern 81, separate follow-up):** `_apply_column_batched_operator` (`optimizer.py:4503/4507`) does an unbounded `jax.vmap(matvec, in_axes=1)(eye)`; byte-gate `_dense_square_operator_materialization_allowed` (`:4510`) bounds OUTPUT not peak JVP working set. Different path (dense-square/Hessian-Newton) than the warm-eval, so it didn't surface; chunk it too if that path is exercised at scale.

## 10. Pointers
- **The plan (next big thing):** `docs/matrix_free_adjoint_gate4_implementation_plan_2026-06-22.md`.
- Completed predecessor task: `HANDOFF-parity-matrix.md` (the 4-seed matrix — DONE, machine precision, commits db14e992e/3c16a7888/223ddb37c).
- Broader arc: `HANDOFF-lbfgs-ondevice-productionization.md`; punch-list `docs/lbfgs_ondevice_open_gates_punchlist.md` (Gate 4/5 — its `surface_objectives_traceable.py` refs are CORRECT; the file exists in `simsopt_jax_adapters/`).
- Pod result jsons: `/tmp/m_fix.json` (the proof: preallocate=true no-OOM + parity), `/tmp/matrix_gpu.json` + `/tmp/matrix_gpu_corner.json` (the 4-seed matrix).
- Reference repos (read-only): optax `/Users/suhjungdae/code/opensource/optax` (HEAD 3205908), torax `/Users/suhjungdae/code/opensource/torax` (HEAD 60190df1).
- Mistake book: `/Users/suhjungdae/.claude/skills/crucible/shared/mistake-book.md` Pattern 81 (incomplete-sibling-chunk) + Pattern 3 occ#3.
- Memory: `project_parity_matrix_completion_2026_06_21`, `project_exact_jacobian_conditioning_measured` (κ≈625), `project_value_and_grad_construction_compile_pole`, `project_ondevice_compile_blowup_root_cause`.
