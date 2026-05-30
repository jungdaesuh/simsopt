# HANDOFF — single-stage GPU E2E parity validation (Plan A follow-through)

> Supersession note, 2026-05-30: this handoff was captured before the broader
> JAX-port remediation commit. The L3 doc/status artifacts referenced below are
> now committed in `f287bde96` (`refactor(jax): close port remediation review`).
> Treat Runpod/result pointers below as historical unless revalidated against the
> current HEAD.

> Last updated: 2026-05-29 09:45 EDT · Status: 3 Runpod runs settled/in-flight. G3 strict ladder DONE (port byte-identical; passed=False = trajectory divergence only). devparity2 (m04) + iota25dev (m10) device-isolation runs RUNNING; pollers armed. Next = read the two device-isolation parity results.

## 1. Goal
Original thread: "test this jax-ported code on runpod, it's cost ~$100/3 days — find the bottleneck and make it finish quicker while maintaining integrity." That resolved to **Plan A** (switch single-stage GPU default `ondevice`→`scipy-jax`, compile-once). Current sub-goal: **validate the single-stage GPU E2E** by (a) picking good, hw-compatible, replayable seeds and (b) proving **CPU↔GPU parity**.
**Definition of done (current):** device-isolation parity numbers (jax-CPU vs jax-GPU) for a fixture seed (m04) and a good converged seed (iota25, m10), plus the strict-ladder verdict. Integrity rule throughout: **never loosen tolerances / fake parity / hide transfer failures**; fix root causes; commit only when asked.

## 2. Where we are right now
Plan A is committed. Seed-selection + replay tooling built & committed. Three Runpod runs were launched to measure CPU↔GPU parity. **G3 (strict ladder) finished**: `passed=False` but for the *right* reason — the JAX objective is **byte-identical** to the C++ oracle at matched candidates (`same_candidate max_*_abs_diff = 0.0`); the failure is pure **trajectory divergence** in a non-converged maxiter=20 run (cpu 4 iters vs jax 20, neither hit iota target). Two **device-isolation** runs (same scipy-jax code on CPU and GPU, so only device FP differs) are still running: `devparity2` (m04 fixture, ~1h ETA) and `iota25dev` (good iota25 seed, m10, ~2-4h ETA). The very next thing is to read those two parity diffs when the pollers fire.

## 3. NEXT ACTIONS (start here on resume)
1. [ ] **Read the device-isolation parity results.** Background pollers: `bul9xyot4` (devparity2 m04), `b1utr1w6s` (iota25dev m10). If this session is gone, query Runpod directly:
   ```bash
   ssh -i /Users/suhjungdae/.runpod/ssh/RunPod-Key-Go -p 16628 -o StrictHostKeyChecking=no root@154.54.102.24 '
   echo "dp2 done: $(cat /root/_devparity2_done.txt 2>/dev/null)"; echo "iota25dev done: $(cat /root/_iota25dev_done.txt 2>/dev/null)"
   for R in devparity2_cpu devparity2_cuda iota25dev_cpu iota25dev_cuda; do
     S=/root/$R; echo "== $R: $(cat $S/exit_code.txt 2>/dev/null||echo RUNNING) =="
     J=$(find $S/out -name results.json -o -name REJECTED.json 2>/dev/null|head -1)
     [ -n "$J" ] && python3 -c "import json;d=json.load(open(\"$J\"));print({k:d.get(k) for k in [\"FINAL_IOTA\",\"FINAL_OBJECTIVE\",\"OPTIMIZER_NFEV\",\"OPTIMIZER_SUCCESS\"]})"
   done'
   ```
   Then diff CPU vs GPU `results.json` (FINAL_IOTA / FINAL_OBJECTIVE / FINAL_NON_QS / FINAL_BOOZER_RESIDUAL). Expectation: tighter than G3 (same code, only device-FP).
2. [ ] Report the two device-isolation diffs to the user (this is the answer to their recurring "per-iter / trajectory CPU-vs-GPU parity" question).
3. [ ] Update memory file `project_autoresearch_seed_replay.md` with the device-isolation numbers.
4. [x] L3 doc/status artifacts were committed in `f287bde96`. Production-maxiter convergence run for a strict-ladder PASS (heavy), `outer_maxls 20→8` consistency, and fresh CUDA signoff remain separate follow-up gates.

## 4. Environment & how to run
- cwd / repo / branch: `/Users/suhjungdae/code/columbia/simsopt-jax` / simsopt-jax / `gpu-purity-stage2-20260405` (remediation HEAD now includes `f287bde96`; older Plan-A commits d06e55b96 + 8cb782525 are in history).
- **Runpod** (A100-80GB, 2TB host RAM): `ssh -i /Users/suhjungdae/.runpod/ssh/RunPod-Key-Go -p 16628 -o StrictHostKeyChecking=no root@154.54.102.24`
  - Deployed src (REPO): `/root/simsopt_e2e_ea597cc14_20260527T191926Z/src_13a664f15_20260528T175259Z` (commit 13a664f15; **no .git** — must `export SIMSOPT_REPO_SHA=$(cat $ROOT/repo_sha.txt)` or provenance/git calls fail).
  - venv python: `/root/simsopt_canary_local/venv/bin/python` (has simsoptpp + jax 0.10).
  - Seeds: `/root/seeds/{iota25_m10n10, iota0064_m8n6, iota148_m8n8, iota25_coldgentle_m10n10, equilibria/}`; m04 fixture seed: `/root/m04seed_gen_20260529T103429Z/out/mpol=4-ntor=4-1c849d10`.
  - Run launch scripts on Runpod: `/root/{g3_now.sh, devparity_full.sh, iota25_devparity.sh, devparity2_{cpu,cuda}.sh, m04_seedgen.sh}`. Markers: `/root/_last_g3_path.txt`, `/root/_devparity2_done.txt`, `/root/_iota25dev_done.txt`.
- **Local validation env** (CLAUDE.md): `PYTHONNOUSERSITE=1 PYTHONPATH=src JAX_ENABLE_X64=True JAX_PLATFORM_NAME=cpu`, python `.conda/jax/bin/python`. `ncdump` available for wout inspection; `netCDF4` NOT in the env.
- **DURABILITY:** always launch Runpod long runs under `tmux`; SSH-foreground dies on disconnect (a confirmed cost leak). All current runs are tmux sessions: `dp2cpu dp2cuda iota25dev`.

## 5. Done so far (with evidence)
- [x] **Plan A committed** — `d06e55b96` (GPU default `ondevice`→`scipy-jax`, `enable_xla_caches="all"`, 2 test updates, repro script, docs). Verified: ruff clean, `tests/test_cli_defaults.py` 8/8.
- [x] **Seed tooling committed** — `8cb782525`: `scripts/select_replayable_seeds.py` (jsonl + `--scan-tree` + `--verify-coil-currents`), `scripts/replay_surrogate_seed.sh` (convert→run, local spec cache). ruff-clean.
- [x] **iota25 at-optimum CPU↔GPU parity** (3-eval, m10): obj |d|=**5.3e-8**, iota 4.6e-5, compiles **179=179** both devices. Verified via results.json diff.
- [x] **iota0064 at-optimum parity** (1-eval, m8): obj |d|=**4e-19**, iota 2.3e-17 (bit-level — it didn't move).
- [x] **Compile-once confirmed** at m04 (177) and m10 (179), CPU and GPU.
- [x] **G3 strict ladder** (m04, C++ vs jax, 2h37m): `passed=False`, but `same_candidate max_objective/gradient/boozer_abs_diff = 0.0` (byte-identical port). Failure = trajectory divergence (cpu 4 iters / jax 20, both OPTIMIZER_SUCCESS=False, iota~0 vs 0.15). Dir `/root/g3_ladder_scipyjax_20260529T102509Z`.
- [x] **L3 CI harness plan** written: `docs/jax_gpu_e2e_ci_harness_plan.md` — committed in `f287bde96`.
- [x] Selector/status edits from the remediation session were committed in `f287bde96`; verify current scope with `git status --short`.
- [~] **devparity2 (m04 fixture) + iota25dev (m10 good seed)** — RUNNING, results pending (NEXT ACTION 1).

## 6. Key decisions & rationale
- **scipy-jax is the GPU lane** (host SciPy L-BFGS outer + one jitted value/grad bundle, compile-once). `ondevice` (one fused monolith) never compiles at m04+ → abandoned. `scipy-jax-fullgraph` = eager op-by-op (CPU default, fine on CPU). Do not relitigate.
- **`--target-lane-boozer-newton-polish-policy run`** is REQUIRED on CUDA for mpol≥6: the default `skip-large-strict-cuda` skips the Newton polish, so the bfgs leg stalls at ~1e-9 and Boozer init fails the 1e-11 gate. `run` re-enables the designed mechanism (NOT a tolerance change). iota16 only "worked" earlier because it was on CPU (skip is CUDA-only).
- **L2 (re-base strict ladder onto scipy-jax) needs NO code refactor.** The ladder CLI already accepts `--optimizer-backend scipy-jax` and resolves the expected outer method per-backend (`single_stage_init_parity.py:3990`). `TARGET_OPTIMIZER_BACKEND`/`DEFAULT_OPTIMIZER_BACKEND` are *on-device* concepts (native-lbfgs group `:553`, etc.) and must stay `ondevice`; the `lbfgs-ondevice` contract gates one separate proof test only. So G3 ran on scipy-jax as-is.
- **Skipped G4** (production-scale): per-step rate ~1.3 min/eval already known from G1; cost. **Skipped iota148**: its results.json lacks `banana_surf_radius` → convert KeyError; coverage already from iota0064+iota25.
- **Gentle perturbation** for cold-starts (iota25dev: 0.3% → ‖dx‖∞=0.004, iota 0.25→0.27, currents stay 80/15.8kA in-envelope). 1% was too aggressive (G1: iota→0.148, hardware-rejected).

## 7. Dead ends / do NOT retry
- **ondevice outer-L-BFGS monolith** — never finishes compiling (>2h, 77GB, 0 steps at m04). Abandoned.
- **Aggressive (1%) perturbation cold-start** (G1) — drove convergence (18 evals) but landed hardware-INFEASIBLE → REJECTED. Use ≤0.3%.
- **Reduced quadrature grid for high mpol** (iota22 at nphi31/ntheta16) — under-resolves → Boozer solve stalls. Keep nphi/ntheta adequate for mpol/ntor.
- **m10 on CPU for long descents** — ~19 min/eval → 4h timeout (C1 cold twin died this way). Use m04 for CPU-feasible device parity; m10 CPU only for short/gentle.
- **L2 contract decoupling refactor** — investigated, **unnecessary AND risky** (would pollute the native-lbfgs group + break 2 test files). Do not do it.
- **Don't flip `DEFAULT_OPTIMIZER_BACKEND` in the smoke fixture** — it's the shared on-device default for many probes.

## 8. Open questions / blockers
- Device-isolation parity numbers (devparity2, iota25dev) — pending (NEXT ACTION 1).
- A **strict-ladder PASS** requires both sides to **converge to the same optimum** (production maxiter, heavy run) — known requirement, not a bug. Not yet run.
- Repo provenance flag (informational): fixture equilibrium `wout_nfp22ginsburg_000_014417_iota15.nc` is **misnamed — it is actually nfp=5** (byte-identical to `wout_nfp5ginsburg_000_014417_iota15.nc`). Doesn't affect parity (both devices read the same file).

## 9. Mental model (hard-won context)
- **Two parity flavors:** (a) *same-candidate* = objective/grad computed at the SAME input → tests the port (should be ~0). (b) *trajectory/end-state* = where the optimizer lands → diverges under FP-chaos when not converged. G3 proved (a)=0.0 but failed (b) because neither side converged at maxiter=20. **The port is correct; the strict end-state gate just needs convergence.**
- **Parity diff scales with how far the run MOVES:** 1 eval (iota0064) → 1e-17; 3 evals (iota25) → 5e-8 obj / 5e-5 iota; many evals → larger (FP accumulation). iota is the loose direction (weakly weighted → flat → larger diff than the minimized objective).
- **Seeds:** good converged optima are the autoresearch `iota25`/`iota0064` (VMEC `000_014417` Ginsburg scan, nfp=5, in-envelope). The repo **fixture** seed (`single_stage_seed_iota15`) is a RAW cold-start test seed (field_error 0.058) — fine for parity-of-machinery, not a physics optimum. All VMEC (version_=9), NOT the `_vmec_legacy_` lineage, NOT DESC.
- **The replay pipeline** (`replay_surrogate_seed.sh`): a seed dir needs `biot_savart_opt.json` + `surf_opt.json` + `results.json`; convert builds a JAX runtime spec (cached locally) from them; results.json must have `TOROIDAL_FLUX`/`banana_surf_radius`/`TF_CURRENT_A`. A REJECTED run writes `REJECTED.json` (with full metrics under `diagnostic_results_payload`) instead of `results.json` — promote the payload if needed.

## 10. Pointers
- Memory (SSOT for findings): `~/.claude/projects/-Users-suhjungdae-code-columbia-simsopt/memory/project_autoresearch_seed_replay.md` and `project_strict_cuda_e2e_cost_bottleneck.md` (indexed in MEMORY.md).
- Earlier-phase continuation: `docs/plan_a_compile_once_continuation_2026-05-29.md` (committed).
- L3 plan: `docs/jax_gpu_e2e_ci_harness_plan.md` (committed in `f287bde96`).
- Replay manifest (local): `/tmp/vacuum_seed_manifest.jsonl`. Selector: `scripts/select_replayable_seeds.py`. Launcher: `scripts/replay_surrogate_seed.sh`.
- Runpod prior parity invocation pattern: `/root/simsopt_e2e_ea597cc14_20260527T191926Z/artifacts/run_r28_ea597cc14.sh`.
