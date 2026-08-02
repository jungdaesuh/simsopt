# Coil47 strict-GPU L-BFGS compile shape

The current custom L-BFGS step-from-start kernel lowered for the FP64
source-owned `coil47` case (`maxcor=10`, `maxiter=2`) on the RTX 5090 in
`2.453 s`. StableHLO was `1,504,879` bytes / `16,172` lines; the JAXPR was
`1,645,584` bytes / `26,985` lines. The lowered graph contained 56 StableHLO
case regions and five while loops.

This is lowering-only diagnostic evidence. Runtime executable count and device
memory were not measured, and the candidate tree was dirty.
