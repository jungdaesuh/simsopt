# RunPod GPU compile-cache validation protocol

Validate that the persistent compilation cache cuts the slow cold XLA:GPU compile
(~73 min on the RunPod cu1290 image, diagnosed "once-slow" in
`docs/jax_scipy_jax_gpu_compile_diagnostic_next.md`) on a **second** run, with no
nvlink regression after the narrow-cache fix (`prepare_cuda_gpu_lowres_tests.py`
`_cuda_env`, commit `0ef2f8f76`).

**Status: this is an operator runbook — it needs a RunPod GPU and your external
artifacts; it has NOT been executed (no GPU locally). Placeholders are `<...>`.**

## Why a network volume is the whole point

The compile cache lands at `<output-dir>/jax_compilation_cache/<label>`. RunPod
pod containers are ephemeral, so the cache only survives — and the cold compile
is only reused — if `<output-dir>` is on a **RunPod network volume** (mounts at
`/workspace`). On the container disk it is lost on every pod restart and you pay
the cold compile every time.

What gets cached (so expectations are right):
- `JAX_COMPILATION_CACHE_DIR` caches the full compiled **executable** on disk —
  the main win; works regardless of the autotune mode.
- `JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=xla_gpu_per_fusion_autotune_cache_dir`
  (the now-safe narrow mode) adds the per-fusion **autotune** cache on top — a
  smaller increment. The broad `"all"` mode (removed) added more but forced
  nvlink through the container toolkit (the cu1290 block).

## Phase 0 — pod + volume (operator)

- Launch a CUDA-12 GPU pod (A100/L40S/etc.).
- **Attach a RunPod Network Volume** → it mounts at `/workspace`. Reuse the *same*
  volume for the warm run / a fresh pod — that is the real ephemeral-pod test.

## Phase 1 — code + deps on the pod

Code sync via git bundle (the proven method; keeps `.git` per repo policy — never
a bare tarball):
```bash
# locally:
git bundle create simsopt-clean.bundle pr/jax-port-clean
#   scp/upload simsopt-clean.bundle to the pod's /workspace
# on the pod:
cd /workspace
git clone simsopt-clean.bundle simsopt-pr-jax-port-clean
cd simsopt-pr-jax-port-clean && git checkout pr/jax-port-clean
python -m pip install --upgrade "jax[cuda12]==0.10.0" etils
python -c "import jax; print(jax.__version__, jax.devices())"   # expect a cuda device
```
(Alternatively clone the fork directly if the pod has credentials.)

## Phase 2 — generate the packet with the cache on the volume

The generator bakes `cd <REPO_ROOT>` into the runner, so run it **on the pod**.
`--output-dir` on `/workspace` is what makes the cache persistent.
```bash
cd /workspace/simsopt-pr-jax-port-clean
python benchmarks/prepare_cuda_gpu_lowres_tests.py \
  --output-dir /workspace/cuda_cache_run \
  --boozer-surface-zip      <YOUR boozer_surface.zip> \
  --autoresearch-runs-dir   <YOUR autoresearch runs dir> \
  --warm-start-run-dir      <YOUR donor warm-start dir> \
  --single-stage-mpol 2 --single-stage-ntor 2     # lowres smoke first; mpol10 for the real 73-min case
```
Required inputs (the generator raises without them): `--boozer-surface-zip`,
`--autoresearch-runs-dir`. Produces `/workspace/cuda_cache_run/run_cuda_gpu_lowres_tests.sh`
and the manifest; the runner's env points `JAX_COMPILATION_CACHE_DIR` at
`/workspace/cuda_cache_run/jax_compilation_cache/<label>`.

Confirm the runner carries the SAFE narrow mode (sanity check the fix landed):
```bash
grep -o 'JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=[^ ]*' /workspace/cuda_cache_run/run_cuda_gpu_lowres_tests.sh | sort -u
# expect: xla_gpu_per_fusion_autotune_cache_dir   (NOT "all")
```

## Phase 3 — COLD run (cache empty), under tmux so a disconnect doesn't kill it

`project_strict_cuda_e2e_cost_bottleneck`: SSH-foreground runs die on disconnect —
always tmux/nohup.
```bash
tmux new -s cudacold
cd /workspace/simsopt-pr-jax-port-clean
/usr/bin/time -v bash /workspace/cuda_cache_run/run_cuda_gpu_lowres_tests.sh \
  2>&1 | tee /workspace/cuda_cache_run/cold.log
# detach with Ctrl-b d ; reattach with: tmux attach -t cudacold
du -sh /workspace/cuda_cache_run/jax_compilation_cache/*   # cache populated
```

## Phase 4 — WARM run (cache hit)

Re-run the SAME runner (cache now on `/workspace`). For the true ephemeral test,
destroy the pod, launch a new one with the SAME network volume, redo Phase 1
deps, then:
```bash
tmux new -s cudawarm
/usr/bin/time -v bash /workspace/cuda_cache_run/run_cuda_gpu_lowres_tests.sh \
  2>&1 | tee /workspace/cuda_cache_run/warm.log
```

## Phase 5 — read the result

- **Decisive metric = wall time of the compile phase, cold vs warm.** The
  persistent cache short-circuits XLA codegen with a disk hit, so the warm run's
  compile wall should drop sharply (the ~73-min cold → a small fraction). Compare
  the `Elapsed (wall clock)` from the two `/usr/bin/time -v` logs, and the JAX
  `Compiling ...` timing lines in cold.log vs warm.log.
- Note: `compile_event_count` / `cache_miss_count` in each case's `results.json`
  (`JAX_COMPILE_DIAGNOSTICS`) count *traces*, which are the SAME cold and warm —
  do not use them for cold-vs-warm. They are the once-vs-recompile signal (closed
  on CPU); here the wall is what moves.
- **nvlink check:** the run must complete on the cu1290 image without the
  `cubin … nvlink` failure. If it still fails cold, the block is independent of
  the cache mode — capture the toolchain dump and escalate separately.

## Decision

- Warm compile wall ≪ cold, run completes nvlink-clean → fix confirmed; for
  production keep `--output-dir` on the network volume so every pod reuses it.
- Warm wall ≈ cold → cache not persisting (check `output-dir` is really on the
  volume and the same volume is reattached) or not hitting (check the cache dir
  populated in Phase 3).

Run lowres (mpol2) first to confirm the *mechanism* cheaply, then mpol10 to
measure the real production cold→warm delta.
