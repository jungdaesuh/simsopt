# simsopt benchmarks

Driven by [airspeed velocity (asv)](https://asv.readthedocs.io/). Tracks kernel-level performance for the fixes in `PERFORMANCE_AUDIT.md`.

## Setup

Run from an active env created by `scripts/dev/env-create.sh`:

```bash
asv machine --yes
```

## First baseline (pre-fix master)

```bash
asv run HEAD^!                    # benchmark the current commit
asv publish && asv preview        # HTML dashboard
```

## Per-PR regression guard

```bash
asv continuous master HEAD        # fail on any regression > 5 %
```

## Benchmarks

| File | Targets | Size labels |
|------|---------|-------------|
| `bench_biotsavart.py` | B1, B2, B4, B5, B6, B10 | prod, small |
| `bench_surface.py`    | A1, A2                 | prod, small |
| `bench_rss.py`        | A1 (peak-RSS guard)    | 100, 500 iters |

Additional benchmarks from `PERFORMANCE_AUDIT.md` §6 (tracing, dipole, full optimizer step) will land as they're needed for their respective fixes.

## Sizes

Every benchmark runs in both **prod** (production-realistic inputs) and **small** (1 coil / 100 points). Small-N catches threading-overhead regressions invisible at prod scale — see `PERFORMANCE_AUDIT.md` §8.6.

## Output fingerprints

Benchmarks that `track_fingerprint(...)` return a scalar norm of their output array. This lets asv flag "benchmark got faster because it now returns zeros" — see `PERFORMANCE_AUDIT.md` §8.6.

## Reference hardware

Benchmark results are only comparable on the same hardware. The current reference is documented in `BASELINE.md` (written on first baseline capture).

## Running subsets

```bash
asv run --bench BiotSavartForward HEAD^!          # one benchmark class
asv run --bench biotsavart HEAD^!                 # one file
asv run -k prod HEAD^!                            # filter by param
```
