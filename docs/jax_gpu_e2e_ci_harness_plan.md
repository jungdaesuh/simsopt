# JAX GPU single-stage E2E — CI/automation harness plan (L3)

Status: **automation plan / stub.** Best implemented after the manual runs
(G1/G2/G3) stabilize so the harness encodes what they actually needed. The
current-head manual CUDA remediation signoff packet is recorded in
`docs/jax_port_review_remediation_plan.md`; this file tracks the future durable
CI runner, committed seed fixtures, and CPU/GPU diff automation.

## Goal
A repeatable, durable gate that runs the single-stage GPU optimization end-to-end
(`scipy-jax` lane) from a known-good in-envelope vacuum seed and asserts:
1. **compile-once** — `compile_event_count` is a small constant, not growing with
   outer steps (the Plan A property);
2. **drives convergence** — a cold (off-optimum) start descends to convergence;
3. **CPU↔GPU parity** — final iota / objective / nonQS / Boozer residual and the
   final coil DOF vector agree within the parity-ladder tolerances;
4. **bounded cost** — wall time and peak memory stay within budget.

## Building blocks (already in the repo)
- `scripts/select_replayable_seeds.py` — pick in-envelope vacuum converged seeds
  (`--scan-tree ... --ready physics --verify-coil-currents`) → replay manifest.
- `scripts/replay_surrogate_seed.sh` — convert→run a seed on cpu/cuda with
  `--record-jax-compile-diagnostics`; spec cached, never mutates the seed dir.
- `benchmarks/single_stage_init_parity.py` / `validation_ladder_contract.py` — the
  strict parity ladder + Tier-3 single-stage proof contract.

## Required pieces (to build)
1. **Seed fixtures committed to the repo.** Pin 1–2 physics-faithful vacuum seeds
   (e.g. an iota25-class m10 seed + a small m8 seed) as test fixtures so CI does
   not depend on `autoresearch/runs/` paths. Each = `biot_savart_opt.json` +
   `surf_opt.json` + `results.json` + the equilibrium `wout_*.nc`.
2. **Cold-start generator.** Deterministic off-optimum perturbation of the free
   coil DOFs (1–2% relative on geometry; leave currents in envelope) so the run
   has a real trajectory. (Prototype used `np.random.default_rng(12345)`.)
3. **A non-interactive runner** wrapping `replay_surrogate_seed.sh` that emits a
   machine-readable summary (compiles, nfev, final metrics, wall, peak mem).
4. **Assertions** against `validation_ladder_contract` tolerances + the compile-once
   invariant + a CPU↔GPU diff step.

## Hard requirements learned from the manual runs
- **Durability:** launch under `tmux`/`nohup` (or a watchdog `setsid`) — SSH
  foreground runs die on disconnect (a confirmed cost leak). CI must detach.
- **Newton polish on CUDA:** large (mpol>=6) CUDA runs need
  `--target-lane-boozer-newton-polish-policy run`, else the Boozer init never
  clears the 1e-11 gate (default `skip-large-strict-cuda` skips it).
- **Resolution/grid:** the Boozer init grid (`nphi/ntheta`) must be adequate for
  `mpol/ntor`; a reduced grid under-resolves and stalls the inner solve.
- **Compile footprint:** the m10 `scipy-jax` bundle compile peaked ~74 GB GPU /
  ~8 GB host (GPU) and ~296 GB host (CPU). CI hardware must budget for this.

## Dependency
The **strict-ladder gate (G3)** depended on **L2**: re-basing the Tier-3 single-
stage proof contract (`validation_ladder_contract.py`
`required_outer_optimizer_method`) and `DEFAULT_OPTIMIZER_BACKEND`
(`benchmarks/single_stage_smoke_fixture.py`) from `ondevice` onto `scipy-jax`,
**without** folding host-SciPy `scipy-jax` into the on-device
`TARGET_NATIVE_LBFGS_OPTIMIZER_BACKENDS` group.

Status: the default rebase has landed in the current worktree. Plain
single-stage and Stage 2 benchmark harness invocations now default their JAX
outer optimizer to `scipy-jax`; `ondevice` remains an explicit target/native
stress lane. Focused parser/proof-contract selectors in
`tests/test_benchmark_helpers.py` and the CUDA proof test definition in
`tests/integration/test_single_stage_physics_parity.py` were updated with this
split. Full CUDA execution remains a separate hardware gate.

## Smallest first step
Wrap the existing `replay_surrogate_seed.sh` cold-start invocation + a CPU twin +
the scalar/DOF diff (already prototyped in the session pollers) into one script
under `scripts/`, runnable locally and from CI, emitting JSON. Promote to a CI
workflow once L2/G3 land.
