# RunPod single-stage benchmark seeds (precision parity · walltime · memory)

Curated, verified seed set for the A100 80GB single-stage benchmark. All seeds
are **complete trios** (`surf_opt.json` + `results.json` + `biot_savart_opt.json`)
with the standard `surf_opt_boozer_surface.json` resume sidecar, **hardware-valid
+ feasible**, sourced from `/Users/suhjungdae/code/columbia/autoresearch/runs/`.
Verified 2026-06-15 (mpol/ntor from `surf_opt.json`; fields from `results.json`).

They are NOT in the repo — they must be **staged onto the pod** (e.g. copied to
`/workspace/seeds/<name>`) since the pod clones only the clean repo.

## The set (2× mpol8 + 4× mpol10; iota 0.046→0.300; vol 0.040/0.045/0.050)

| # | dir (under `autoresearch/runs/`) | mpol/ntor | iota | vol | schema | validity standout |
|---|---|---|---|---|---|---|
| 1 | `edge_envelope_vacuum_2026-06-09/seeds/vac_aws0150` | 8/8 | 0.0459 | 0.0500 | v1 | hw-valid, feasible, nested |
| 2 | `edge_envelope_vacuum_2026-06-09/seeds/m9_onspec_aws0150` | 8/8 | 0.1085 | 0.0499 | v1 | hw-valid, feasible, nested |
| 3 | `april_signed_donor_probe_2026-05-21/flipped_mirror_iota285_seed` | 10/10 | 0.2850 | 0.0399 | v1 | **OPTIMIZER_SUCCESS=true**, confinement 1.0 |
| 4 | `loop_2026-05-18/flipNV2_iota298_cw_label_preserved` | 10/10 | 0.2979 | 0.0399 | legacy | **Poincaré 50/50** (verified) |
| 5 | `april_signed_donor_probe_2026-05-21/flipped_mirror_iota299_seed` | 10/10 | 0.2998 | 0.0399 | v1 | OPTIMIZER_SUCCESS=true, confinement 0.956 |
| 6 | `prod0142_edge_vacuum_typekk_zero_contact_iota004_vol003/radius_ladder_kchamp_021_to_0142/seed_kchamp_cw_with_state` | 10/10 | 0.2835 | 0.0449 | legacy | distinct basin, extra boozer_state sidecar |

All: NFP=5, vacuum, R0≈0.976, TF=-80000 A, 30 coils. `OPTIMIZER_SUCCESS=false` on
#1/#2/#4/#6 is benign (best-feasible/iter-limit snapshot); the physics gates
(feasibility, hardware, nesting, topology) all pass.

## Coverage by test axis

- **Precision parity** (CPU cpp-reference vs JAX-GPU): all 6 are physics-valid, so
  the comparison is meaningful, not garbage-in. Operating-point spread (iota
  0.046→0.300, vol 0.040/0.045/0.050) stresses parity across the physics space;
  #4 is the Poincaré-gold reference point.
- **Walltime**: 2× mpol8 vs 4× mpol10 gives compile + solve time scaling with
  resolution.
- **Memory**: mpol8 vs mpol10 gives peak host RSS + device-memory scaling; the
  mpol10 seeds approach the production magnitude (the ~73-min-compile regime).

## How to run each seed (on the A100)

`single_stage_init_parity.py` is the parity harness: it runs the CPU cpp
reference lane AND the JAX/GPU lane, and records timings + MaxRSS (+ GPU memory
via the launcher's nvidia-smi). One run per seed = one parity + walltime + memory
data point:
```
PYTHONPATH=src JAX_ENABLE_X64=1 python benchmarks/single_stage_init_parity.py \
  --platform cuda --optimizer-backend scipy-jax \
  --warm-start-run-dir /workspace/seeds/<seed-name> \
  --record-jax-compile-diagnostics \
  --case-artifacts-dir /workspace/bench/<seed-name> \
  --output-json /workspace/bench/<seed-name>.json
```
- **Parity** = `comparison`/`failures` (final iota/volume/field-error CPU-vs-JAX
  relative diffs) in the output JSON.
- **Walltime** = `/usr/bin/time -v` Elapsed + the JSON `timings`.
- **Memory** = MaxRSS (host) + the GPU `memory.used` CSV.
- Persist `--case-artifacts-dir`/`--output-json` under `/workspace` (network
  volume) so they survive the pod, alongside the compile cache.

## Caveats
- **#4 / #6 use the legacy results.json schema** (`CONTRACT_SCHEMA_VERSION=null`).
  The runtime seed spec is *compiled* schema-1 from the donor regardless, but
  confirm the warm-start loader accepts the legacy results.json on the first
  seed before the full sweep. #1/#2/#3/#5 are `CONTRACT_SCHEMA_VERSION=1`
  (lowest-risk).
- **No mpol2 seeds exist** in `autoresearch/runs` — the cheap mpol2 *mechanism*
  check (cache hit / nvlink-clean) still uses the clean-repo fixture via
  `init-parity --mpol 2` (self-compiled seed); these donors are mpol8/mpol10 for
  the production-scale parity/walltime/memory numbers.
