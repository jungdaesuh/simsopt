# Coil47 custom JAX versus Optax strict-GPU fixed-budget comparison

Custom JAX and Optax L-BFGS used the same FP64 `coil47` fixture and two-step
budget on the strict CUDA lane. Neither converged (`status=1` for custom;
Optax exposes no status). Final objectives differed by `2.96e-10` absolute
(`2.15e-9` relative), and final parameters differed by `8.94e-5`.

Warm times were `0.162 s` custom and `21.338 s` Optax; solver RSS deltas were
`546292` and `444792 KiB`. This is a fixed-budget diagnostic comparison, not
an endpoint or promotion certificate. The candidate tree was dirty and the
Optax 20-step qualification remains open.
