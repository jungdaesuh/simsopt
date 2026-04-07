# Single-Stage Replay Validation (2026-04-08)

This note records manual replay checks against archived rows from
`/Users/suhjungdae/code/opensource/autoresearch/results_surrogate.jsonl`
using the current `simsopt-surrogate` checkout and the repo-local `.venv`.

## Setup

- Current code path:
  `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py`
- Local equilibria root:
  `/Users/suhjungdae/code/columbia/DATABASE/EQUILIBRIA`
- Replay environment:
  `OMP_NUM_THREADS=2 PYTHONPATH=src ./.venv/bin/python`

## Row 465

- Archived row: `465`
- Equilibrium: `nfp5_iota21`
- Archived Stage 2 seed:
  `/Users/suhjungdae/code/opensource/autoresearch/single_stage_results/outputs-wout_nfp5ginsburg_desc_iota17.nc/mpol=8-ntor=6-01dbcea4-1775565158882/biot_savart_opt.json`
- Local output root:
  `/Users/suhjungdae/code/columbia/simsopt-surrogate/tmp/e2e_replay_row465`
- Local results artifact:
  `/Users/suhjungdae/code/columbia/simsopt-surrogate/tmp/e2e_replay_row465/mpol=8-ntor=6-d3476e33/results.json`

Replay verdict: reproduced cleanly. Final metrics matched the archived row to
floating-point noise.

| metric | archived | replay |
| --- | --- | --- |
| `FIELD_ERROR` | `0.0003081476791931617` | `0.00030814767919315955` |
| `OBJECTIVE_J` | `0.16217264745643137` | `0.16217264745643203` |
| `FINAL_IOTA` | `0.16993075988035486` | `0.16993075988035478` |
| `FINAL_VOLUME` | `0.09997905941467236` | `0.09997905941467224` |
| `CURVE_CURVE_MIN_DIST` | `0.050000000819843474` | `0.050000000819843474` |
| `MAX_CURVATURE` | `36.15714661695133` | `36.15714661695133` |

Observed optimizer behavior:

- The live replay explored several rejected candidates before returning to the
  archived incumbent.
- Rejections included self-intersecting surfaces and hardware-invalid points.
- This did not change the final replay verdict because the accepted terminal
  point matched the archived result.

## Row 470

- Archived row: `470`
- Equilibrium: `nfp5_iota21`
- Archived Stage 2 seed:
  `/Users/suhjungdae/code/opensource/autoresearch/single_stage_results/outputs-wout_nfp5ginsburg_desc_iota21.nc/mpol=8-ntor=6-d3476e33-1775578133118/biot_savart_opt.json`
- Local output root:
  `/Users/suhjungdae/code/columbia/simsopt-surrogate/tmp/e2e_replay_row470_retry`
- Local run directory:
  `/Users/suhjungdae/code/columbia/simsopt-surrogate/tmp/e2e_replay_row470_retry/mpol=8-ntor=8-72e70296`

Replay verdict: numerically reproduced the archived incumbent, but the script
did not terminate cleanly within the validation window.

Archived final values:

- `field_error = 0.00017016898201770319`
- `objective_J = 0.15826455005757206`
- `final_iota = 0.17028763498648902`
- `final_volume = 0.0999924937170206`
- `curve_curve_min_dist = 0.050000000819843335`
- `curve_surface_min_dist = 0.07236108016287608`
- `surface_vessel_min_dist = 0.0866511464603457`
- `max_curvature = 36.1571466169515`

Observed live replay evidence before manual stop:

- The initialization solve hit the archived incumbent exactly to displayed
  precision:
  - `iota from solve: 0.17028763498648886`
  - `Volume: 0.09999249371702072`
- Later rejected candidates followed the same pattern as row 465:
  - self-intersecting surfaces
  - hardware-invalid candidates with spacing/curvature violations
- The line search then shrank back to the archived incumbent:
  - `Iota: 0.17028763498648886`
  - `Volume: 0.09999249371702072`
  - step sizes dropped to the `1e-17` to `1e-18` range

Termination note:

- No `results.json` was written before manual stop.
- The process appeared stuck in a vanishing step-size loop at the archived
  incumbent rather than exiting with a final termination message.
- This is a replay/termination issue, not evidence of a refactor mismatch in
  the recovered solution itself.

Monolith cross-check:

- The historical monolith entrypoint from commit
  `721073a85ab87cc5017705e77aa0593f568387dc` was rerun from
  `/Users/suhjungdae/code/columbia/simsopt-surrogate/tmp/anchor_worktree_721073a8`
  with the same row 470 inputs while still importing the current fixed
  `src/simsopt` tree.
- The monolith reproduced the same sequence of rejected candidates, restored
  the same incumbent
  (`Iota = 0.17028763498648886`, `Volume = 0.09999249371702072`), and then
  entered the same vanishing step-size loop.
- Therefore the row 470 termination stall is inherited monolith behavior, not
  a regression introduced by the modularized refactor.

## Takeaway

For the `nfp5_iota21` ladder checked so far:

- Row 465 is a full clean reproduction.
- Row 470 reproduces the archived solution numerically, but the current run
  path appears to have a termination-loop issue after restoring that solution.
