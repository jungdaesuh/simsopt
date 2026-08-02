# `coil47` GPU five-sample comparison

Candidate: `3b2b9f40ab46c34c0087cde434dc73f145d14588`
Device: strict RTX 5090 CUDA, FP64
Samples: one discarded warm-up plus five retained samples per provider

| Provider | Cold median (s) | Warm median (s) | Warm range (s) | Max solver RSS (KiB) | Max process VRAM (MiB) | Iterations |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Custom | 14.43294 | 0.044042 | 0.040160--0.046399 | 1,786,248 | 1,514 | 12--13 |
| Optax | 4.07767 | 0.027639 | 0.024482--0.029384 | 1,605,436 | 2,602 | 15 |

Custom/Optax ratios: `3.54x` cold, `1.59x` warm, and `1.11x` maximum solver
RSS. The custom maximum VRAM was `0.58x` the Optax maximum in this sample.
Process-attributed VRAM was sampled from `nvidia-smi` over the runner and its
provider child; it is not total device occupancy.

Raw warm samples:

```text
custom: 0.046399, 0.040160, 0.044042, 0.044489, 0.042271 s
optax:  0.024482, 0.025953, 0.027639, 0.029384, 0.027790 s
```

Retained VRAM samples:

```text
custom: 1514, 1480, 1514, 1480, 1480 MiB
optax:  1472, 2602, 1472, 1472, 1472 MiB
```

All ten endpoints succeeded. Final-objective differences across providers
were at most `5.55e-17`; custom used 12--13 iterations and Optax used 15.
Every raw runner payload reports the candidate commit and a clean worktree.
The receipt and raw archive are
`coil47-custom-optax-gpu-five-sample-vram-20260802`.

This is RTX-5090 evidence, not A100 qualification.
