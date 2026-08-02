# Coil47 fixed-budget strict-GPU diagnostic

The current custom JAX L-BFGS implementation ran the source-owned FP64
`coil47` fixture for two steps on the RTX 5090. Native CPU comparison matched
the final objective exactly, the gradient infinity norm within
`2.2e-17`, and final parameters within `2.2e-16`.

Cold time was `81.30 s` and warm time `0.162 s`; fixture setup took `8.74 s`.
Solver RSS peaked at `1,939,584 KiB` with a `546,292 KiB` solver delta. The
large cold value is compile/initialization cost, not steady-state optimizer
time. This is a fixed-budget diagnostic, not convergence or promotion
evidence: the worktree was dirty, status was 1, and no matched GPU Optax or
A100 run was performed.
