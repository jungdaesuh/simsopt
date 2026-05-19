# SIMSOPT JAX Port Status

Slide-friendly brief for SIMSOPT physics users.

Date: 2026-05-15

---

## One-Slide Summary

**What is working**

- Major smooth physics kernels are now available on the JAX path.
- CPU C++/SciPy vs JAX CPU parity is passing for the covered fixed-state research fixtures.
- Current matrix: **27 fixtures = 21 pass / 6 partial / 0 fail**.
- Supported comparisons: **251 CPU/JAX comparisons pass**.

**What is still open**

- Real CUDA/GPU parity is not closed on the local Apple Silicon machine.
- The 6 partial rows are explicit host/external/discrete workflow boundaries, not numerical failures.
- Full release acceptance needs a clean-HEAD CUDA artifact.

---

## Why This Matters For Physics Users

The port is intended to support:

- faster repeated field and objective evaluations,
- differentiable objective construction,
- same-state CPU/JAX verification before GPU use,
- GPU acceleration for smooth fixed-state optimization kernels,
- stronger regression checks for optimization workflows.

The goal is not to rewrite every SIMSOPT script. The goal is to move the
physics-heavy smooth compute path to JAX while keeping host orchestration,
external solvers, plotting, and file output outside the target lane.

---

## Validation Chain

The acceptance chain is:

```text
SIMSOPT CPU C++/SciPy oracle
        -> JAX CPU
        -> JAX GPU
        -> JAX CPU/GPU agreement
```

Current status:

| Link | Status |
| --- | --- |
| CPU C++/SciPy -> JAX CPU | Good for covered rows |
| CPU C++/SciPy -> JAX GPU | Open |
| JAX CPU -> JAX GPU | Open |
| Full CUDA release stamp | Open |

---

## Current Evidence Snapshot

Latest matrix used for this brief:

```text
.artifacts/parity/20260514-partial-closeout/all-fixtures.json
```

| Metric | Value |
| --- | ---: |
| Fixtures | 27 |
| Passing fixtures | 21 |
| Partial fixtures | 6 |
| Failing fixtures | 0 |
| Supported CPU/JAX comparisons | 251 |
| JAX backend in artifact | CPU |
| GPU status | runtime required |

Interpretation:

**Covered CPU/JAX precision parity is good. GPU parity still needs hardware.**

---

## Example File Coverage

The current matrix covers 26 unique Python example files through 27 fixture
rows. One example has two fixture rows:

```text
examples/3_Advanced/curves_CWS_example.py
  - cws_saved_local_flux_nfp2
  - cws_saved_local_flux_nfp3
```

Coverage against the main tutorial example buckets:

| Example bucket | Covered files | Total files |
| --- | ---: | ---: |
| `examples/1_Simple` | 8 | 11 |
| `examples/2_Intermediate` | 14 | 26 |
| `examples/3_Advanced` | 4 | 8 |
| Main tutorial examples | 26 | 45 |

If every Python file under `examples/` is counted, including support modules,
banana helper libraries, plotting utilities, and benchmark drivers, the same
matrix covers 26 of 77 files.

---

## Covered Example Files

| Example file | Fixture status |
| --- | --- |
| `examples/1_Simple/optimize_coil_position_orientation.py` | pass |
| `examples/1_Simple/permanent_magnet_simple.py` | pass |
| `examples/1_Simple/qfm.py` | partial |
| `examples/1_Simple/stage_two_optimization_minimal.py` | pass |
| `examples/1_Simple/surf_vol_area.py` | pass |
| `examples/1_Simple/tracing_fieldlines_NCSX.py` | pass |
| `examples/1_Simple/tracing_fieldlines_QA.py` | pass |
| `examples/1_Simple/tracing_particle.py` | pass |
| `examples/2_Intermediate/boozer.py` | pass |
| `examples/2_Intermediate/boozerQA.py` | pass |
| `examples/2_Intermediate/permanent_magnet_MUSE.py` | pass |
| `examples/2_Intermediate/permanent_magnet_PM4Stell.py` | pass |
| `examples/2_Intermediate/permanent_magnet_QA.py` | partial |
| `examples/2_Intermediate/stage_two_optimization.py` | pass |
| `examples/2_Intermediate/stage_two_optimization_finite_beta.py` | pass |
| `examples/2_Intermediate/stage_two_optimization_planar_coils.py` | pass |
| `examples/2_Intermediate/strain_optimization.py` | pass |
| `examples/2_Intermediate/tracing_boozer.py` | partial |
| `examples/2_Intermediate/wireframe_gsco_modular.py` | pass |
| `examples/2_Intermediate/wireframe_gsco_sector_saddle.py` | pass |
| `examples/2_Intermediate/wireframe_rcls_basic.py` | partial |
| `examples/2_Intermediate/wireframe_rcls_with_ports.py` | partial |
| `examples/3_Advanced/coil_forces.py` | pass |
| `examples/3_Advanced/curves_CWS_example.py` | pass |
| `examples/3_Advanced/stage_two_optimization_finitebuild.py` | pass |
| `examples/3_Advanced/wireframe_gsco_multistep.py` | partial |

---

## Physics Workflows Covered

| Workflow area | JAX status | Evidence style |
| --- | --- | --- |
| Biot-Savart field evaluation | Covered | CPU C++/JAX fixed-state parity |
| Flux objective / B dot n | Covered | `SquaredFlux` / `integral_BdotN` parity |
| Stage-II coil optimization ingredients | Covered for reduced fixed states | example-derived objective and gradient rows |
| Curve length and coil-distance penalties | Covered | CPU/JAX objective component parity |
| Curvature penalties | Covered | CPU/JAX objective component parity |
| Surface area, volume, toroidal flux | Covered | CPU surface-objective parity |
| Boozer residual and labels | Covered for fixed states | CPU/JAX residual and scalar-label parity |
| Permanent-magnet fixed-state algorithms | Covered for reduced examples | PM grid, moments, residuals, histories, fields |
| Wireframe fixed-state RCLS/GSCO | Covered for reduced examples | matrices, constraints, fields, histories |
| Fieldline / particle tracing endpoints | Covered as output parity | endpoint, time/status, hit-count parity |

---

## What "Covered" Means

For this brief, "covered" means:

- the CPU lane is the independent SIMSOPT CPU/C++/SciPy behavior,
- the JAX lane is built independently through JAX specs/kernels,
- the same fixed state is evaluated on both lanes,
- active DOF identity is checked before gradient comparison,
- numerical differences are checked against named tolerance buckets,
- the row is not counted from JAX-vs-JAX self-consistency alone.

This is acceptance/regression evidence for named research fixtures. It is not a
formal theorem over all possible inputs.

---

## Precision Picture

| Quantity | Current observation |
| --- | --- |
| Worst absolute difference | `1.668088953010738e-5` |
| Where it occurs | tracing particle endpoint |
| Tolerance lane | `event_time_tracing` |
| Worst relative difference | `1.9296860502983134e-4` |
| Why acceptable | near-zero value with roundoff-scale absolute error |

Smooth objective lanes use tighter tolerances:

| Lane | Typical tolerance |
| --- | --- |
| direct kernels | `rtol=1e-10`, `atol=1e-12` |
| derivative-heavy rows | `rtol=1e-8`, `atol=1e-10` |
| event/tracing endpoints | `rtol=1e-6`, `atol=1e-8` |

---

## Test Coverage Snapshot

| Test inventory | Count |
| --- | ---: |
| Total Python test files | 214 |
| JAX/parity-named test files | 90 |
| Non-integration JAX/parity files | 84 |
| Integration JAX/parity files | 6 |

Important integration files:

- `tests/integration/test_stage2_jax.py`
- `tests/integration/test_single_stage_jax_cpu_reference.py`
- `tests/integration/test_single_stage_physics_parity.py`
- `tests/integration/test_non_banana_example_cpp_jax_cpu_parity.py`

Recent focused validation:

- QFM/current focused slice: **7 passed, 3 skipped**
- Broader QFM/flux related slice: **50 passed, 31 skipped**

---

## Guard Against Toy Or Tautological Evidence

Current controls:

- CPU/JAX matrix rows require numeric `cpu_cpp_value` and `jax_cpu_value`.
- CPU/JAX gradient rows check active DOF basis alignment.
- `simsoptpp` is required for the CPU oracle in the non-banana parity harness.
- Oracle lint rejects JAX-vs-JAX comparisons as CPU/C++ parity proof.
- Self-consistency tests are allowed only as self-consistency tests.

Bottom line:

**The counted matrix rows are not based on toy-only or JAX-vs-JAX evidence.**

---

## Current Passing Fixture Families

Examples represented in passing rows:

- Stage-II minimal and composite coil optimization fixtures
- planar and full Stage-II examples
- finite-beta target-flux fixture
- finite-build multifilament fixture
- CWS local-flux saved fixtures
- surface area/volume fixture
- Boozer residual and Boozer QA scalar wrappers
- permanent-magnet simple, MUSE, and PM4Stell reduced fixtures
- wireframe GSCO modular and sector/saddle reduced fixtures
- tracing QA/NCSX fieldline and particle endpoint fixtures
- strain optimization support gate
- coil force and magnetic-energy support gate

---

## Six Partial Rows

These are not failed numerical comparisons.

| Partial row | Why partial |
| --- | --- |
| QFM surface | fixed-state QFM pieces pass; full host `QfmSurface` solve orchestration is not claimed |
| RCLS basic wireframe | fields/matrices pass; raw current vector is non-unique because of nullspace |
| PM QA | fixed-state relax-and-split pieces pass; coil-current optimization and output writing remain host-side |
| RCLS with ports | same nullspace issue as basic RCLS, with port constraints preserved |
| GSCO multistep | first-step diagnostic passes; full mutation/pruning/final-adjust/output loop is host/discrete workflow |
| Boozer guiding-center tracing | cached-state endpoint path passes; VMEC/BOOZXFORM input execution is external-solver boundary |

---

## Differentiability Boundary

Good JAX/autograd targets:

- smooth field evaluation,
- B dot n and flux objectives,
- surface and curve scalar objectives,
- fixed-state Boozer residuals and labels,
- fixed-state PM and wireframe objective components,
- smooth fixed-time pieces of tracing.

Not currently a differentiable product path:

- event/status hit logic,
- mutation/pruning workflows,
- raw output/report writing,
- external VMEC/BOOZXFORM/SPEC execution,
- generic SciPy/MPI host orchestration.

---

## GPU Acceptance Still Needed

What must be produced on CUDA hardware:

1. Clean-HEAD full parity matrix.
2. CPU C++/SciPy vs JAX GPU comparison.
3. JAX CPU vs JAX GPU comparison.
4. Exact provenance:
   - git SHA,
   - clean/dirty tree state,
   - JAX and jaxlib versions,
   - `JAX_ENABLE_X64=1`,
   - CUDA/runtime metadata,
   - GPU model,
   - exact command lines.

Until this exists, CUDA status remains open.

---

## Recommended Next Steps

**Immediate**

- Regenerate the full matrix from current clean HEAD.
- Run the same matrix on real CUDA hardware.
- Archive the CUDA artifact with provenance.

**Only if product need is confirmed**

- Consider differentiable `QfmSurface` solve layer.
- Consider PM QA coil-current optimization in JAX.
- Treat differentiable event tracing as a separate research problem.

**Do not spend effort on**

- plotting/output-only paths,
- VMEC/BOOZXFORM/SPEC execution as a simple JAX port,
- broad generic `Optimizable`/SciPy/MPI replacement without a target workflow.

---

## Message To Physics Users

The JAX port is now useful for many smooth fixed-state SIMSOPT physics
calculations and optimization ingredients.

The strongest current statement is:

```text
For covered research fixtures, CPU C++/SciPy and JAX CPU agree within the
declared precision lanes, with no supported comparison failing.
```

The remaining release question is not whether to add more toy tests. It is:

```text
Can the same evidence be reproduced on real CUDA hardware with full provenance?
```
