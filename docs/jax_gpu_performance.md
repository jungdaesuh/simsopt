# JAX Port — GPU Performance and Numerical Parity

This document records the measured performance and numerical fidelity of the JAX
single-stage optimization backend relative to the C++ (`simsoptpp`) reference, across
NVIDIA GPU generations. It is intended for reviewers evaluating the port.

All measurements use a fixed single-stage objective evaluation (objective value `J`
*and* gradient `dJ`) at a matched, resolved Boozer state, so the C++ and JAX lanes do
identical physical work and the comparison is apples-to-apples.

## Summary

- **Correctness:** the JAX backend reproduces the C++ reference to machine precision —
  the objective value is bit-identical and the gradient agrees to `|dJ|` ≈ 2.8×10⁻¹⁴
  (relative L2). Parity holds across C++, JAX-CPU, and JAX-GPU.
- **Performance:** JAX-GPU is competitive with the C++ reference on Ampere (A40) and
  **faster than C++ on Hopper** (H100/H200), at machine-precision parity.
- **Hardware:** the dominant cost is **double-precision (FP64) compute**, not memory
  bandwidth. Gains track the GPU's FP64 throughput, so Hopper-class cards are strongly
  preferred for single-solve latency.
- **Precision:** **double precision is required.** The forward Boozer solve tolerates
  single precision, but the adjoint (gradient) linear solve is numerically singular in
  FP32. The backend runs in FP64 by default.

## 1. Benchmark configuration

| Parameter | Value |
| --- | --- |
| Field periods (NFP) | 5 |
| Boozer poloidal/toroidal resolution | `mpol = ntor = 10` |
| Surface quadrature | `nphi = 127`, `ntheta = 32` |
| Target rotational transform | `iota* = 0.285` |
| Target volume | `0.04` |
| Modular (TF) coils | 20 |
| Precision | float64 |
| Quantity measured | one single-stage `J + dJ` evaluation, wall-clock |

Two timings are reported:

- **Cold** — includes one-time XLA compilation (JAX) / first-call setup.
- **Warm** — a re-evaluation at a perturbed design point, which forces a *full* Boozer
  re-solve. This is the realistic per-outer-step cost during an optimization, and is the
  primary number; it exceeds the cold time for every lane (C++ included) because it does
  the full nonlinear solve without the cold run's initial-guess advantage.

The absolute numbers below are specific to this configuration; the *ratios* are the
portable result.

## 2. Numerical parity (correctness)

At the matched resolved state, the JAX objective is **bit-identical** to C++ and the
gradient matches to machine precision:

| Comparison | Objective `J` | Gradient `dJ` (rel. L2) |
| --- | --- | --- |
| C++ ↔ JAX-CPU | bit-identical | ≈ 10⁻¹⁶ |
| C++ ↔ JAX-GPU | bit-identical | ≈ 2.8×10⁻¹⁴ |

The small GPU-side `dJ` residual is the expected consequence of non-associative
floating-point reduction order on the GPU, not an algorithmic difference.

## 3. Single-evaluation performance (cross-GPU)

Warm per-evaluation wall-clock, JAX-GPU vs. the C++ reference on the same node:

| GPU | JAX-GPU | C++ | JAX vs. C++ |
| --- | --- | --- | --- |
| A40 (Ampere)  | 270.6 s | 220.7 s | 1.23× slower |
| H100 (Hopper) | 113.0 s | 128.7 s | **1.14× faster** |
| H200 (Hopper) | 107.6 s | 122.3 s | **1.14× faster** |

JAX-GPU crosses over from slower-than-C++ on Ampere to faster-than-C++ on Hopper. The
A40 → H100 speedup for the JAX lane is **2.4×**. H200 ≈ H100 (within ~5%), because the
peak device footprint for this problem is only ~0.34 GiB — far too small to benefit from
H200's higher memory bandwidth (see §6).

## 4. Production optimization lane

The production single-stage lane (`scipy-jax-decomposed`: a host-driven SciPy L-BFGS-B
outer loop over a decomposed device value/gradient kernel) measured on H100:

| Metric | H100 | A40 |
| --- | --- | --- |
| Cold initial evaluation | 110.9 s | ~412 s |
| Warm steady-state per evaluation | ~32 s | — |

The A40 → H100 cold speedup is **3.7×**. The cold-to-warm gap (110.9 s → ~32 s on H100)
is XLA compilation amortized across the optimization; steady-state per-evaluation cost is
~32 s, so a multi-step production run averages near that figure rather than the cold time.

## 5. Multi-GPU throughput

Data-parallel evaluation of independent configurations (one configuration pinned per GPU)
scales near-linearly, because each solve's small device footprint avoids contention:

| Configuration | Speedup | Efficiency |
| --- | --- | --- |
| 4 × H100 | 3.79× | 95% |
| 4 × H200 | 3.90× | 97.5% |

Multi-GPU buys **throughput** (more configurations per unit time), not lower single-solve
latency.

## 6. Hardware selection guidance

The performance lever is **FP64 compute throughput**, not memory bandwidth:

- The single-stage gradient is dominated by an FP64-heavy adjoint linear solve.
- The peak device footprint (~0.34 GiB) is far below the bandwidth-bound regime, so the
  H200's higher HBM bandwidth yields no gain over the H100 (the two are within ~5%).
- Gains track the card's FP64 ratio: the A40 (GA102) runs FP64 at 1:32 of FP32
  (~1.2 TFLOP/s), whereas Hopper runs 1:2 (~34 TFLOP/s). This is why the JAX lane is slower
  than C++ on the A40 but faster on Hopper.

**Recommendation:** for single-solve latency, use a Hopper-class (H100/H200) or other
high-FP64 GPU; an A40 or other 1:32-FP64 card is not competitive. For many independent
configurations, scale out across GPUs (§5). To fully exploit the backend, batch/`vmap`
over configurations (throughput) rather than expecting one small FP64 solve to out-latency
tuned C++ on a low-FP64 card.

## 7. Precision requirements (double precision is required)

Single precision was evaluated and is **not viable for gradient-based optimization**. The
two stages of a single-stage evaluation have very different conditioning:

- **Forward Boozer solve (finding the surface):** FP32-tolerant. In FP32 it converges to
  the same surface as FP64 (rotational transform agrees to ~9 significant digits), at ~4%
  lower wall-clock and half the device memory. The forward solve is launch/bandwidth-bound
  here (it converges immediately from the warm-started state), so FP32 yields little speed.
- **Adjoint linear solve (the gradient):** **fails in FP32.** The adjoint solves a linear
  system whose conditioning is the square of the Boozer Jacobian conditioning (see
  Appendix). In FP64 this is well-conditioned and yields the machine-precision gradient of
  §2. In FP32 the system is numerically singular — the factorization produces a non-finite
  (NaN) result, so no gradient can be formed and the optimizer cannot take a step.

Because the expensive, accuracy-critical work lives in the adjoint, and the adjoint is
exactly the part that fails in FP32, the backend requires FP64. The only meaningful FP32
opportunity is *mixed precision* (FP32 for the well-conditioned forward field evaluations,
FP64 for the solve and adjoint), which is not currently implemented.

## 8. Backend modes

The runtime backend is selected by the `SIMSOPT_BACKEND_MODE` environment variable
(`src/simsopt_jax/backend/runtime.py`):

| Mode | Engine / device | Precision | Use |
| --- | --- | --- | --- |
| `native_cpu` | C++ / CPU | float64 | reference implementation |
| `jax_cpu_parity` | JAX / CPU | float64 | bit-for-bit CPU parity checks |
| `jax_cpu_fast` | JAX / CPU | float64 | fast CPU (fused kernels) |
| `jax_gpu_parity` | JAX / CUDA | float64 | GPU parity (the benchmarked GPU mode) |
| `jax_gpu_fast` | JAX / CUDA | float64 | fast GPU (fused kernels) |
| `jax_cpu_float32_smoke` | JAX / CPU | float32 | shape/wiring smoke tests only — not for production accuracy (see §7) |

## 9. Reproduction

The cross-lane evaluation matrix is produced by `benchmarks/single_stage_objective_parity_matrix.py`,
which evaluates the C++ and JAX lanes at the same resolved state and prints the wall-clock
comparison.

```bash
# JAX-GPU (float64) lane
JAX_ENABLE_X64=1 \
SIMSOPT_BACKEND_MODE=jax_gpu_parity SIMSOPT_JAX_PLATFORM=cuda \
python -m benchmarks.single_stage_objective_parity_matrix \
    --seeds <seed_run_dir> --equilibria-dir <equilibria_dir> \
    --lane-label jax-gpu --out matrix_gpu.json \
    --mpol 10 --ntor 10 --nphi 127 --ntheta 32 \
    --jax-materialize-dense-linearization false
```

`--jax-materialize-dense-linearization false` skips a reporting-only dense linearization
to bound warm-evaluation peak memory; it does not affect the gradient. Each `--seeds`
directory is an optimization seed containing `results.json`, `biot_savart_opt.json`, and
`surf_opt_boozer_surface.json`; `--equilibria-dir` holds the corresponding `wout_*.nc`
equilibria. The script also reports the C++ and JAX-CPU lanes for the parity check of §2.

The production lane is run via the single-stage example with
`--optimizer-backend scipy-jax-decomposed` under the same `jax_gpu_parity` backend mode.

## Appendix: measured conditioning

For this configuration, the exact Boozer Jacobian is well-conditioned, with condition
number κ ≈ 625. The adjoint solves the associated Gauss–Newton (normal-equations) system,
whose condition number is the square, κ ≈ 3.9×10⁵. The threshold for a finite, accurate
direct solve is roughly `machine-epsilon < 1/κ`:

- **float64** (ε ≈ 2.2×10⁻¹⁶): κ·ε ≈ 9×10⁻¹¹ — ~9 orders of margin; the solve is
  accurate to ~10 digits and the gradient matches C++ to machine precision.
- **float32** (ε ≈ 1.2×10⁻⁷): κ·ε ≈ 5×10⁻² — no margin; the system is numerically
  singular and the solve returns a non-finite result.

This is the quantitative basis for the FP64 requirement in §7.
