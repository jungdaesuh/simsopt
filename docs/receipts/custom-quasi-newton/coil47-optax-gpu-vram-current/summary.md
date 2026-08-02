# Coil47 Optax strict-GPU VRAM measurement

The FP64 Optax two-step fixture completed on `cuda:0` with the platform
allocator and preallocation disabled. Warm time was `22.147 s`; solver RSS
delta was `444348 KiB`. The sampled provider process reached `834 MiB` of
VRAM (`2.56%` of `32607 MiB`).

The two-step endpoint was capped (`stopping_reason=iteration-limit`), and the
candidate tree was dirty. This is memory/fixed-budget diagnostic evidence,
not convergence or promotion evidence.
