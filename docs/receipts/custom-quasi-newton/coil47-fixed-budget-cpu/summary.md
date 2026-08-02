# Coil47 fixed-budget provider comparison

Native SIMSOPT, custom JAX L-BFGS, and Optax L-BFGS used the same FP64
source-owned `coil47` fixture and two-step budget on CPU. Native and custom
matched the final objective to `2.8e-17` absolute and the final parameters to
`3.1e-14` maximum absolute difference. Optax differed from native by
`3.0e-10` in objective and `8.9e-5` in parameters at the same cap.

Warm times were `0.523 s` native, `0.079 s` custom, and `7.994 s` Optax;
solver RSS deltas were `0`, `336144`, and `410400 KiB`, respectively. All
results are fixed-budget diagnostics: native/custom reported status 1, Optax
reported no status, and the candidate worktree was dirty. This is not a
convergence or promotion receipt.
