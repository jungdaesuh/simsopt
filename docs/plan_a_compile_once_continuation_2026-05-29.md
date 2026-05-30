# Plan A continuation handoff (2026-05-29)

Read this first after compaction. It is self-contained. Companion memory file:
`~/.claude/projects/-Users-suhjungdae-code-columbia-simsopt/memory/project_strict_cuda_e2e_cost_bottleneck.md`
(has the full evidence trail). Approved plan: `~/.claude/plans/plan-a-vectorized-dragonfly.md`.

## TL;DR — where we are RIGHT NOW

- We diagnosed why the strict-CUDA single-stage E2E burned ~3 days / ~$100 on Runpod: **compilation architecture, not physics**. The GPU default lane `ondevice` compiles the whole outer L-BFGS + objective into ONE XLA monolith that **never finishes compiling** at m04 (>2 h, 77 GB host RSS, 0 outer steps). The objective math is correct (same-candidate parity ~1e-12; inner-solve tolerances byte-identical to C++).
- **Plan A** = make the single-stage GPU objective compile **once** and reuse it across the host-SciPy outer loop, by switching the GPU default from `ondevice` to `scipy-jax` (host SciPy L-BFGS-B outer + on-device inner Boozer solve + jitted target-lane value/grad bundle).
- Staged diagnostic proved **Case I (cheap win)**: `scipy-jax` already compiles-once (CPU probe: `compile_event_count=175` at BOTH maxiter=12 and 24 → maxiter-invariant) and converges (m24 → `OPTIMIZER_SUCCESS=True`, "CONVERGENCE"). So no objective restructure needed.
- **Code change is DONE + CPU-validated** (uncommitted). **Step 4 (GPU confirmation) is RUNNING on Runpod** as of ~01:30 UTC 2026-05-29.

## NEXT ACTION (do this first)

Check the Step 4 GPU run result. Background waiter task id: **`bdn3k6oc0`** (output file under the session tasks dir). Or query Runpod directly:

```bash
ssh -i /Users/suhjungdae/.runpod/ssh/RunPod-Key-Go -p 16628 -o StrictHostKeyChecking=no root@154.54.102.24 '
S=$(cat /root/_last_s4_path.txt); tmux ls 2>&1|grep s4; cat "$S/exit_code.txt" 2>/dev/null
grep -iE "Elapsed \(wall|Maximum resident" "$S/time.stderr" 2>/dev/null
find "$S/out" -name jax_compile_diagnostics.json|head -1|xargs -r python3 -c "import json,sys;d=json.load(open(sys.argv[1]));print(\"compiles\",d[\"compile_event_count\"])"
J=$(find "$S/out" -name results.json -o -name REJECTED.json|head -1); echo $(basename $J)
python3 -c "import json;d=json.load(open(\"$J\"));print({k:d.get(k) for k in [\"FINAL_IOTA\",\"OPTIMIZER_SUCCESS\",\"OPTIMIZER_STATUS\",\"TERMINATION_MESSAGE\"]})"
'
```

The S4 run: `scipy-jax`, m04 (mpol=4,ntor=4,nphi=63,ntheta=32), maxiter=25, CUDA, warm-started, `--record-jax-compile-diagnostics`. Live status at last check (~22 min in): **GPU at 81% util** (monolith was 0%), **RSS 3.6 GB** (monolith 77 GB), past compile, running the outer optimizer — i.e. the architecture works. Awaiting final: total wall, compile_event_count (expect small constant ≈175-class), iota progress toward 0.15, exit code.

**Interpreting Step 4:**
- Completes + small constant compile count + GPU busy → Plan A CONFIRMED end-to-end. Done with Step 4.
- Note: `outer_maxls` resolved to **20** on GPU (not the tighter target-lane 8) — minor config-consistency gap (more line-search/step; slower not wrong); also the cause of 3 pre-existing test failures (see Open items).
- If it stalls/OOMs → capture forensics (watchdog writes heartbeat.log); but compile-once is already proven on CPU, so a GPU failure would more likely be a different issue (memory policy, a specific m04 trial point), not the architecture.

## THE CODE CHANGE (uncommitted — `git diff` to see; commit ONLY when user asks)

Files modified (tracked):
1. `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py`
   - `resolve_single_stage_default_optimizer_backend` (~line 8005): JAX branch `return "scipy-jax"` for both CPU and GPU/CUDA. The previous platform split is removed.
   - `--optimizer-backend` help text (~line 4466): updated to say the JAX default is now `scipy-jax` (compile-once), `ondevice` must be selected explicitly.
2. `tests/test_cli_defaults.py`: 2 assertions for cuda+jax default `ondevice` → `scipy-jax` (lines ~56 and ~119/180 region; the `test_resolve_*` and `test_single_stage_parse_args_uses_platform_default`).
3. `tests/geo/test_single_stage_example.py`: renamed `test_parse_args_defaults_jax_backend_to_ondevice_optimizer_lane` → `..._to_scipy_jax_optimizer_lane`; assertion now `== module.resolve_single_stage_default_optimizer_backend("jax")`.
4. `src/simsopt/backend/runtime.py` (~line 2387, from EARLIER this session): added `jax.config.update("jax_persistent_cache_enable_xla_caches", "all")` in the cache-config block — complements Plan A (persists GPU kernel cache so the one-time compile survives across processes). Validated by import smoke (111 passed) + backend tests (128 passed).

Also created (untracked, NOT part of the core change but useful):
- `scripts/single_stage_target_lane_repro.sh` — parametrized target-only reproducer (PLATFORM/OPTIMIZER_BACKEND/MPOL/… env vars; uses the bootstrap-wrapper + correct env; durability note re tmux).
- `docs/jax_test_debugging.md` — appended a runbook section on the cheap target-lane debug loop.

## VALIDATION DONE (CPU, no GPU)
- ruff check clean on all touched files; touched lines format-clean (pre-existing unrelated format drift in test files left untouched — do NOT reformat whole files, it churns unrelated lines).
- `tests/test_cli_defaults.py` 8/8 pass; parse-args defaults 3/3 pass.
- **Zero regression proven**: full `test_single_stage_example.py` (public lane) had 372 passed / 3 failed; the 3 failures (`outer_maxls_to_tighter_budget`, `benchmark_mode_preserves_target_lane_boozer_precision`, `defaults_boozer_algorithm_from_explicit_inner_backend`) are **PRE-EXISTING CPU-env failures** — proven by `git stash` of my edits + re-running on clean HEAD (identical failures). They’re GPU-env tests that fail on any CPU-only checkout.

## OPEN / FOLLOW-UP ITEMS
1. **Step 4 final metrics** (in progress) — record wall time, compile count, convergence; update the memory file + tell the user.
2. **`outer_maxls=20` vs 8 for scipy-jax**: scipy-jax IS in `_JAX_TARGET_OUTER_OPTIMIZER_BACKENDS`/`_JAX_TARGET_OUTER_MAXLS_BACKENDS` (single_stage_banana_example.py ~189-200) but still resolves maxls=20 on GPU — the maxls resolver has a platform/condition gate that excludes scipy-jax from the tighter 8. Decide: make scipy-jax get the tighter target-lane defaults (and fix the 3 pre-existing tests), or accept 20. This is the likely real follow-on.
3. **Full CPU-vs-GPU convergence parity at m04** (heavier; needs the ~76-min C++ reference) — the real "does GPU reach an equally-valid optimum" test. Deferred; gated on Plan A being in.
4. **Commit** — not done (commit only when user asks). When asked: scope to the 4 tracked files (+ optionally the script/doc); end commit msg with the Co-Authored-By Claude Opus 4.8 line; branch is `gpu-purity-stage2-20260405` (already a feature branch).
5. **Strict-CUDA-E2E ladder realignment**: the recovery plan (`docs/strict_cuda_e2e_runpod_recovery_plan_2026-05-28.md`) tested `ondevice`. Since `ondevice` is abandoned as the default (can't compile), the ladder/signoff should be re-based on `scipy-jax`. Separate doc/process change, not Plan A code.

## KEY FACTS / ENVIRONMENT
- Runpod: `ssh -i /Users/suhjungdae/.runpod/ssh/RunPod-Key-Go -p 16628 root@154.54.102.24`. Root: `/root/simsopt_e2e_ea597cc14_20260527T191926Z`. venv py: `/root/simsopt_canary_local/venv/bin/python`. Deployed source: `src_13a664f15_20260528T175259Z` (commit 13a664f15; scipy-jax LANE is identical to local — Plan A only changed the default selector + tests, not the lane).
- Marker files on Runpod: `/root/_last_s4_path.txt` (Step 4), `/root/_last_parity_path.txt`, `/root/_last_warm_path.txt`, etc.
- Local validation env (CLAUDE.md): `PYTHONNOUSERSITE=1 PYTHONPATH=src JAX_ENABLE_X64=True JAX_PLATFORM_NAME=cpu`, python `.conda/jax/bin/python`. JAX 0.10.0.
- m04 warm-start donor (Runpod): `.../m04n04_i05_current_13a664f15_20260528T175502Z/case_artifacts/seed_outputs/mpol=4-ntor=4-66b0718f`.
- DURABILITY: always launch Runpod long runs under `tmux`/`nohup` — SSH-foreground runs die on disconnect (a confirmed earlier cost leak). The watchdog pattern (setsid heartbeat) survives session kill.

## KEY EVIDENCE (compressed)
- ondevice monolith m04 maxiter=5: SIGTERM at 2h (RC=143), still compiling, 77 GB RSS, GPU 0%, 0 outer steps.
- scipy-jax-fullgraph m04 maxiter=5: 26.5 min, eager op-by-op (1170 compiles/20 evals on CPU). NOT the bundle — uses M5 Optimizable composite (`_strict_scalar_value_and_grad`, jax.vjp fresh closure/call).
- scipy-jax (THE FIX) CPU m02: compile_event_count=175 at maxiter=12 AND 24 (compile-once), converges (m24 SUCCESS). The 171 cache-misses are one-time warm-up (inner BFGS/line-search + hardware-penalty closures "re-defined repeatedly") — maxiter-invariant, so NOT per-step; optional future polish to hoist those closures, but not required.
- remat (P1) was a no-op (forward-only loop); traced-maxiter (P2) was defeated by `m=min(maxcor,maxiter)` workspace shape — both rejected before implementing. The real fix was the lane switch, not optimizer surgery.
