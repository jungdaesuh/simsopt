# Coil47 custom JAX strict-GPU comparison

Custom JAX L-BFGS on the strict CUDA lane converged in 12 iterations / 15
evaluations (`status=0`) on the FP64 source-owned `coil47` fixture. Against the
matched native CPU receipt, final objective difference was `5.55e-17`, maximum
final-parameter difference `2.94e-15`, and final gradient-infinity-norm
difference `4.61e-18`.

Fixture build was `7.773 s` with `1,228,756 KiB` peak RSS. Cold/warm solver
times were `75.169/0.354 s`; solver RSS delta was `509964 KiB`. The run used
CUDA-only JAX with preallocation disabled and the platform allocator. The
candidate tree was dirty, so this is diagnostic evidence, not promotion
evidence; Optax and A100 lanes remain open.
