# Coil47 custom L-BFGS strict-GPU VRAM measurement

The FP64 custom L-BFGS run converged in 12 iterations / 15 evaluations with
status `0`. Cold/warm solver time was `78.599 s / 0.253 s`; solver RSS delta
was `524468 KiB`.

During the runner lifetime, `nvidia-smi` sampled the runner parent and provider
child every 0.2 seconds. Peak runner-process GPU memory was `1076 MiB` of
`32607 MiB` (`3.30%`). Desktop compute processes were present separately, with
a `635 MiB` peak; they are not included in the runner figure.

Against the fresh native CPU reference, final objective difference was
`2.78e-17`, gradient-infinity-norm difference `5.37e-11`, and maximum parameter
difference `1.03e-9`. This is dirty-tree diagnostic evidence; it is not an
A100 or Optax qualification receipt.
