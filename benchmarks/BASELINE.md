# Baseline benchmark numbers

Reference point for every "before/after" claim in the fixes planned by `PERFORMANCE_AUDIT.md`. All subsequent PRs compare against this commit.

## Hardware / config

| Property | Value |
|----------|-------|
| Machine  | Apple MacBook Pro, M3 Max |
| OS       | Darwin 25.2.0 |
| Cores    | 14 (perf+efficiency mix) |
| RAM      | 36 GiB |
| OMP_NUM_THREADS | 8 (pinned for reproducibility) |
| Python   | 3.11 (conda-forge, in ./env) |
| XSIMD    | enabled (`simsoptpp.using_xsimd == True`) |
| Baseline commit | `8b63ab7e` |
| Baseline date | 2026-04-19 |

## Numbers

### BiotSavart forward (prod: 16 coils × 200 quad × 10 000 eval points)

| Derivs | Wall time |
|--------|-----------|
| 0 (B)      | 6.71 ± 0.09 ms |
| 1 (+dB)    | 21.3 ± 0.2 ms |
| 2 (+ddB)   | 52.7 ± 0.7 ms |

### BiotSavart forward (small: 1 coil × 50 quad × 100 eval points)

| Derivs | Wall time |
|--------|-----------|
| 0 | 32.1 ± 4 μs |
| 1 | 28.7 ± 2 μs |
| 2 | 58.7 ± 2 μs |

### BiotSavart VJP (prod)

| Path | Wall time |
|------|-----------|
| `B_vjp`        | 36.7 ± 0.2 ms |
| `B_and_dB_vjp` | 82.8 ± 1 ms |

### BiotSavart VJP (small)

| Path | Wall time |
|------|-----------|
| `B_vjp`        | 52.0 ± 3 μs |
| `B_and_dB_vjp` | 102 ± 1 μs |

### Surface VJP (SurfaceRZFourier, mpol=ntor=8, 64×64 quadpoints)

| Call | prod | small |
|------|------|-------|
| `dgamma_by_dcoeff_vjp`     | 229 ± 1 μs | 53.4 ± 2 μs |
| `dgammadash1_by_dcoeff_vjp` | 268 ± 3 μs | 53.6 ± 2 μs |
| `dgammadash2_by_dcoeff_vjp` | 231 ± 3 μs | 51.7 ± 1 μs |

### Peak RSS over N consecutive surface VJP calls

| N iters | Peak RSS (MiB) |
|---------|----------------|
| 100 | 218 |
| 500 | 217 |

Note: the leak from A1 is ~24 bytes × threads per call at `small` size (`num_dofs ≈ 25`); below RSS measurement noise here. The benchmark is still useful once A1 is "fixed" — regression is then any *growth* at prod-scale surfaces. For catching A1 itself, use the ASan build (`scripts/dev/build.sh asan && scripts/dev/test.sh`).

### Output fingerprints (regression guard — stable scalars)

| Benchmark | Size | Derivs | Fingerprint |
|-----------|------|--------|-------------|
| `BiotSavartForward` | prod  | 0 | 196.0643728610337 |
| `BiotSavartForward` | prod  | 1 | 197232.95123707538 |
| `BiotSavartForward` | prod  | 2 | 364510178.40386397 |
| `BiotSavartForward` | small | 0 | 3.6257001770430164 |
| `BiotSavartForward` | small | 1 | 783.0579511309847 |
| `BiotSavartForward` | small | 2 | 279842.01944977173 |

Any post-fix PR that changes these by more than `rtol=1e-12` must either justify (genuine numerical improvement) or is a bug.

## Reproducing

```bash
conda activate ./env
OMP_NUM_THREADS=8 asv run
asv publish && asv preview
```

## What's not yet benchmarked

Future benchmarks to add as their respective fixes approach:

- Particle tracing (B3) — needs C++-level parallel tracing and GIL release to be meaningful.
- Dipole field (C3) — needed before any cache-tiling prototype lands.
- Full optimizer step (a combined forward + gradient loop) — integrates all the biotsavart + surface benchmarks.
- Thread-count scan (1, 2, 4, 8 threads) on the existing benchmarks — exposes the coils-vs-points-parallelism issue (B1) as the actual scaling curve.
