# Coil47 Optax strict-GPU 20-step qualification

The bounded Optax run did not produce a solver payload. Its provider child
exceeded the declared 120-second watchdog and exited after `TERM`. This is
incomplete evidence, not a convergence, parity, or performance result.

The run used FP64, CUDA, the platform allocator, and the current dirty
checkout. The two-step diagnostic remains the only strict-GPU Optax result.
