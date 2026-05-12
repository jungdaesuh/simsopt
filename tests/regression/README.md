# Colleague-artifact regression panel

Forward-pinned regression tests that assert the simsopt math layer
(`src/simsopt/`, `src/simsoptpp/`) produces stable numerical outputs at fixed
configurations supplied by a collaborator's optimization artifacts.

Plan and design rationale:
[`docs/regression_panel_colleague_artifacts_2026-05-11.md`](../../docs/regression_panel_colleague_artifacts_2026-05-11.md).

## Scope and acceptance

This panel is a **local Darwin/arm64 forward gate**, not a CI gate. The
acceptance line is:

```sh
OMP_NUM_THREADS=1 python -m pytest tests/regression/ -q
```

CI integration is intentionally out of scope (see plan §5.5).

## What this panel proves

For each of the 4 colleague artifacts (`bsurf_opt_{01,02,10,20}kA.json`,
finite-`I` current scan, generated on Perlmutter from simsopt
`fc208e657`/2026-04-06):

- The artifact deserializes at HEAD via the examples-side
  `BoozerFiniteIDecoder` bridge loader.
- Surface geometry, Volume, Biot-Savart `B` and `dB`, first-coil curve `γ`
  and `dγ/dc`, and the `CurveCurveDistance` objective reproduce
  snapshotted values to per-category ULP-level tolerance.
- The Path-B Boozer kernel matches the finite-`I` wrapper at
  `G_eff = G₀ + iota·I`, with `iota ≠ 0`. Witness assertion checks that
  `kernel(G_eff) ≠ kernel(G₀)` — the finite-`I` term is not silently
  dropped.
- Biot-Savart `B` is exactly linear in coil currents (bit-equality oracle,
  no snapshot).
- Biot-Savart cache returns bit-equal `B` after a no-op mutation round
  trip (cache-invalidation oracle, no snapshot).

## What this panel does NOT prove

- ALM, frontier, or basin-hopping outer-loop behavior. They live under
  `examples/single_stage_optimization/banana_opt/` and are not invoked here.
- HW-spec audit (`banana_opt/hardware_*.py`) compliance. Independent
  concern.
- The physically-correct `BoozerResidual.J()` / `dJ()` — that requires a
  solved-state sidecar that the artifacts do not currently carry. The
  Path-B kernel test exercises the C++ kernel numerics, not the solved
  residual. See plan §3.1 Path A for the upgrade path.

## What this panel is platform-pinned to

The snapshots are pinned to the host that ran
`_generate_colleague_snapshots.py`. This panel was generated on:

- **Darwin ARM64 / Apple Silicon**
- **numpy with Accelerate BLAS**
- `OMP_NUM_THREADS=1`

Cross-platform reproducibility is an explicit non-goal. If CI runs on a
second platform (Linux x86_64), generate a platform-keyed second snapshot
and parametrize by platform. Do not relax tolerances.

## Files

| File | Purpose |
|---|---|
| `_helpers.py` | Shared utilities (SHA, eval-point generators, leaf-current walker, env summary). Imported by both the scripts and the pytest module. |
| `_smoke_colleague_artifacts.py` | Diagnostic script — prints every quantity for one or more artifacts. Writes nothing. Use when debugging numerics drift. |
| `_generate_colleague_snapshots.py` | Snapshot writer. Re-evaluates the invariants and writes one JSON per artifact under `colleague_artifact_snapshots/`. |
| `test_colleague_artifact.py` | Pytest module. Parametrized over the 4 currents. Asserts SHA + per-category tolerance against the stored snapshot. |
| `colleague_artifact_snapshots/bsurf_opt_*.snapshot.json` | Frozen baseline. **Treat as immutable.** Regeneration requires reviewer sign-off and a justification entry (see plan §11 R3). |

## Running

```sh
# Run the panel
OMP_NUM_THREADS=1 python -m pytest tests/regression/test_colleague_artifact.py -v

# Diagnostic dump (no writes)
OMP_NUM_THREADS=1 python tests/regression/_smoke_colleague_artifacts.py

# Regenerate snapshots (only after justified intentional math change)
OMP_NUM_THREADS=1 python tests/regression/_generate_colleague_snapshots.py
```

The pytest module skips automatically if either the colleague artifacts
or the snapshot files are missing.

## When tests fail

Failures point at a specific invariant on a specific artifact, e.g.
`test_biot_savart_eval[10kA]`. The diagnostic-prefix fields in the
snapshot (`*_sample_first10_flat`) give the first 10 entries for triage,
even though the primary check is the SHA. To investigate:

1. Run the smoke script to print the current values alongside the
   snapshot.
2. `git log --oneline -- src/simsopt src/simsoptpp` since the snapshot's
   `head_sha` (in `_meta`) — the failing invariant likely localizes to a
   commit touching the relevant file.

## Adding a new artifact

1. Place the JSON under the artifact directory (currently outside the
   repo at `/Users/suhjungdae/code/columbia/banana_drivers/inputs/`).
2. Add the key to `_helpers.ARTIFACT_KEYS`.
3. Run `_generate_colleague_snapshots.py`. Commit the new snapshot file
   alongside.
4. The pytest module parametrizes over `ARTIFACT_KEYS` automatically.
