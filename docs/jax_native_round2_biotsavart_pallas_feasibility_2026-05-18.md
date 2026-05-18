# Biot-Savart Pallas/Triton Feasibility Decision

Date: 2026-05-18

## Verdict

No Pallas/Triton product rewrite is approved from the current evidence.

The local run is CPU-only, so it cannot prove a CUDA HBM bottleneck. The
accepted action is to keep the current XLA Biot-Savart kernels and require a
real CUDA value/VJP memory profile plus parity and AD proof before any custom
kernel becomes an implementation item.

## Probe

Command:

```bash
python benchmarks/biotsavart_pallas_feasibility_probe.py \
  --platform cpu \
  --ncoils 8 \
  --nquad 64 \
  --npoints 256 \
  --warmup 1 \
  --repeat 3 \
  --output-json /tmp/biotsavart_pallas_feasibility_probe_moderate.json
```

Observed backend: CPU, one CPU device.

## Current XLA Profile

| Kernel | min time (s) | mean time (s) | temp bytes | arg bytes | output bytes | HLO lines |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `biot_savart_B` | 0.000338 | 0.000476 | 5,242,888 | 30,784 | 6,144 | 439 |
| `biot_savart_B_vjp` | 0.000643 | 0.000705 | 15,763,464 | 36,928 | 24,664 | 1,235 |

Estimated largest current-kernel intermediate arrays for this shape:

| Intermediate | Estimated bytes |
| --- | ---: |
| `point_minus_gamma` | 3,145,728 |
| `cross_gammadash_residual` | 3,145,728 |
| `weighted_integrand` | 3,145,728 |
| `squared_distance` | 1,048,576 |
| `inverse_radius_cubed` | 1,048,576 |

## Decision Boundary

- CPU evidence records the current XLA behavior and the intermediate-size
  pressure points.
- CUDA feasibility remains unproven on this machine.
- No custom-kernel prototype was started because the prerequisite CUDA
  bottleneck evidence is absent.
- A future proposal must compare value, VJP/gradient, memory, and runtime
  against the current XLA kernels before entering product code.
