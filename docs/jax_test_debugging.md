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
