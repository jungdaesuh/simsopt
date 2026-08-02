# Coil47 native/custom/Optax CPU 20-step comparison

All three providers reached the same final objective to recorded precision
and reported success. Native and custom both took 13 iterations; Optax took
16 and does not expose SciPy-style status or evaluation counts here.

Custom versus native: final objective difference `2.78e-17`, final gradient
infinity-norm difference `5.37e-11`, and maximum parameter difference
`1.03e-9`. Optax versus native: objective difference `0`, gradient difference
`6.40e-10`, and maximum parameter difference `1.32e-4`.

Warm time / solver RSS delta were `1.910 s / 0 KiB` native,
`0.171 s / 354368 KiB` custom, and `49.584 s / 1548672 KiB` Optax. This is
matched CPU diagnostic evidence on a dirty checkout; it is not GPU/A100
promotion evidence and does not claim identical trajectories.
