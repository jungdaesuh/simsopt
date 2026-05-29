# JAX Test Debugging

Use the live runner when debugging full-suite failures or slow shards:

```bash
scripts/run_pytest_live.sh
scripts/run_pytest_live.sh tests/integration/test_single_stage_jax.py -m single_stage -x
scripts/run_pytest_live.sh tests/integration/test_stage2_jax.py -m stage2 -x
```

What it does:
- streams pytest output in real time
- writes a timestamped full log under `.artifacts/pytest/`
- writes JUnit XML for machine-readable failure summaries
- enables `--durations=100` so the slowest tests are reported
- enables live Stage 2 subprocess output by setting `SIMSOPT_STAGE2_TEST_STREAM_LOGS=1`

Useful shards:

```bash
python -m pytest tests -m "not slow"
python -m pytest tests -m single_stage
python -m pytest tests -m stage2
python -m pytest tests -m boozer
python -m pytest tests -m integration
```

Stage 2 direct profiling entrypoints:

```bash
python examples/single_stage_optimization/STAGE_2/banana_coil_solver.py \
  --backend jax \
  --probe-only \
  --profile-step-json /tmp/stage2_step_profile.json

python examples/single_stage_optimization/STAGE_2/banana_coil_solver.py \
  --backend jax \
  --record-warm-timings \
  --maxiter 1
```

## Debugging the `lbfgs-ondevice` target lane on GPU (cheap loop)

`benchmarks/single_stage_init_parity.py` is the **signoff parity wrapper**. For
every rung it runs four lanes serially on the same box: seed -> C++ CPU
reference -> JAX same-candidate replay -> JAX target. The C++ reference alone is
~76 min at m04 and runs at **0% GPU utilization** (it needs no GPU), so using the
full parity wrapper as the *debug loop* for the JAX target lane re-pays ~1.5 h of
non-target work before the lane under test even starts. Do not debug the target
lane that way.

To isolate just the production `lbfgs-ondevice` outer optimizer against a
precomputed warm-start donor, use:

```bash
REPO=$PWD \
PYTHON_BIN=/path/to/venv/bin/python \
DONOR=/path/to/prior_run/case_artifacts/seed_outputs/mpol=4-ntor=4-XXXX \
PLATFORM=cuda MPOL=4 NTOR=4 NPHI=63 NTHETA=32 MAXITER=1 \
bash scripts/single_stage_target_lane_repro.sh
```

This runs *only* the JAX target lane (no CPU reference, no replay). It is a debug
tool, not a signoff path — oracle/parity comparison still belongs to
`single_stage_init_parity.py`.

Pitfalls the script encodes for you:

- **Set `SIMSOPT_BACKEND=jax`** (env), not just `--backend jax` (CLI). The CLI
  flag drives the example's own logic, but `get_backend_config()` reads the
  `SIMSOPT_BACKEND` env var (default `cpu`); without it the run silently falls to
  the OOM-prone ondevice-on-CPU path.
- **Import order**: simsopt must import before jax (to resolve GPU-memory env),
  or `_assert_jax_not_imported_for_gpu_memory_config` raises. The script launches
  the example through the same `bootstrap_local_simsopt()` `-c` wrapper the parity
  harness uses.
- **Durability**: launch long remote runs under `tmux`/`nohup`. A job started in
  an SSH foreground gets SIGHUP'd on disconnect and dies mid-run (the tell is an
  empty `/usr/bin/time -v` output file), wasting hours of compute:

  ```bash
  tmux new-session -d -s repro 'bash scripts/single_stage_target_lane_repro.sh'
  ```

Reading the result: a host RSS that balloons to tens of GB while GPU utilization
sits at 0% means the monolithic `jax.jit(run)` graph is compiling/serializing on
the host (the m04 stall signature). A run that allocates GPU memory and then
shows GPU activity before returning a final `results.json` is healthy.

