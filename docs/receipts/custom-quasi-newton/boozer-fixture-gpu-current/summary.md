# Boozer traceable fixture — strict GPU diagnostic

The source-owned FP64 Boozer fixture passed on `cuda:0` with the exact
traceable Newton route. The test took `126.55 s` (`130.66 s` process elapsed)
and reached `2,631,308 KiB` peak RSS.

The run includes the nested residual JIT and rematerialization changes. It is
fixture-construction evidence only: the outer BFGS solve, native comparison,
StableHLO size, and device-memory counter were not collected. The direct probe
used a 180-second bound; it is not a promotion receipt.
