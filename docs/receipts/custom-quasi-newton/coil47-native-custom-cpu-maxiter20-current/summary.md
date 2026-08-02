# Coil47 converged native/custom CPU comparison

Native SIMSOPT and custom JAX L-BFGS used the same FP64 source-owned `coil47`
fixture with a 20-iteration cap. Both converged (`status=0`): native/custom
took 12/13 iterations and 15/44 evaluations. Final objectives were identical
to reported precision (`0.137862632844302`); the maximum final-parameter
difference was `8.10e-15`, and final gradient-infinity-norm difference was
`3.96e-18`.

Cold/warm times were `4.129/1.685 s` native and `18.708/0.279 s` custom.
Solver RSS deltas were `0` and `383736 KiB`. This is a dirty-tree diagnostic
receipt, not a promotion receipt; strict-GPU and A100 comparisons remain open.
