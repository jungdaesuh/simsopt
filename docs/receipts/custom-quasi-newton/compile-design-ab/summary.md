# Physical compile design A/B comparison

The same 47-variable FP64 quadratic, `maxcor=10`, and two-step budget were
used for both L-BFGS compile designs. Design A (the old generic transition)
reached lowering and timed out at 120.48 s with 1,676,884 KiB peak RSS and no
payload. Design B (the specialized fixed-shape transition) completed in
10.35 s, lowered in 2.37 s, and used 420,228 KiB peak RSS with a 1,180,046
byte StableHLO payload.

This isolates compile/control-graph cost from the objective and supports
continuing with design B. It is diagnostic evidence, not a promotion receipt:
the candidate tree was dirty and the comparison was CPU-only.
