# HBT_BANANA Weighted-Only Port

This is an experimental copy of the banana Stage 2 and singlestage workflow for
checking BoozerSurface optimization in this baseline-original checkout.

V1 is weighted-only. It does not import or call an augmented Lagrangian solver,
and it patches only script-level incompatibilities with this SIMSOPT baseline.

## Smoke Commands

From the repository root:

```bash
cd examples/single_stage_optimization/HBT_BANANA
BANANA_STAGE2_MAXITER=20 python jhalpern30/stage2.py 0.0
BANANA_MPOL=2 BANANA_NTOR=2 BANANA_SINGLESTAGE_MAXITER=2 BANANA_SINGLESTAGE_MAXLS=2 python jhalpern30/singlestage.py outputs/I0.0kA/biot_savart_opt.json --current-kA 0.0
```

Use a Python environment with this SIMSOPT baseline and the compiled
`simsoptpp` extension installed; `PYTHONPATH` alone is not sufficient.

Stage 2 writes `biot_savart_opt.json` only when L-BFGS-B reports optimizer
success. Budget-limited or failed runs write `biot_savart_failed.json` and exit
nonzero; failed artifacts are for inspection, not downstream singlestage input.
The singlestage smoke uses low surface resolution so it reaches the L-BFGS-B
path quickly after a successful Stage 2 handoff. A nonzero exit from a Boozer or
SciPy solver failure is expected for this viability check; import/path/API
failures are not.

## Outputs

By default outputs are written under:

```text
examples/single_stage_optimization/HBT_BANANA/outputs/
```

Set `BANANA_OUT_DIR=/path/to/output` to override that directory.

## Scope

The `jhalpern30/` scripts are the direct smoke path. The top-level
`02_stage2_driver.py` and `03_singlestage_driver.py` are retained for the
copied pipeline shape and compile checks; they still use the copied run registry
and require configured parent run IDs before normal pipeline execution.
