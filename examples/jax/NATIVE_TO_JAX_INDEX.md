# Native-to-JAX example index

Generated from `manifest.json`, `parity_manifest.json`, and `authority_evidence.json`. Do not edit this table by hand.

Latest authority evidence:

- `bounded`: pass; run `20260729T005942Z-5ade9aee`; 26 cases / 78 lanes / 1,248 comparisons.
- Evidence revision: `11340c829690fdc0652e47588f5da549829c056a`; summary SHA-256: `fa235cacb0f3e4fa7abc6e8ff4b2f888b2e20c7392bd1531eeb983abad67d66a`; scope: `local_only`.
- `native_default`: not run.

| Native example | JAX mirror | Classification | Runtime dependencies | Device scope | Scale | Latest evidence |
| --- | --- | --- | --- | --- | --- | --- |
| `examples/1_Simple/just_a_quadratic.py` | `examples/jax/1_Simple/just_a_quadratic.py` | eligible / mirror | none | cpu: full_workflow, gpu: full_workflow | bounded | pass (`20260729T005942Z-5ade9aee`) |
| `examples/1_Simple/logger_example.py` | — | not_applicable | none | — | — | not run |
| `examples/1_Simple/minimize_curve_length.py` | `examples/jax/1_Simple/minimize_curve_length.py` | eligible / adapter | none | cpu: jax_region, gpu: jax_region | bounded | pass (`20260729T005942Z-5ade9aee`) |
| `examples/1_Simple/permanent_magnet_simple.py` | `examples/jax/1_Simple/permanent_magnet_simple.py` | eligible / mirror | none | cpu: full_workflow, gpu: full_workflow | bounded | pass (`20260729T005942Z-5ade9aee`) |
| `examples/1_Simple/qfm.py` | `examples/jax/1_Simple/qfm.py` | eligible / adapter | none | cpu: jax_region, gpu: jax_region | bounded | pass (`20260729T005942Z-5ade9aee`) |
| `examples/1_Simple/stage_two_optimization_minimal.py` | `examples/jax/1_Simple/stage_two_optimization_minimal.py` | eligible / adapter | none | cpu: jax_region, gpu: jax_region | bounded | pass (`20260729T005942Z-5ade9aee`) |
| `examples/1_Simple/surf_vol_area.py` | `examples/jax/1_Simple/surf_vol_area.py` | eligible / adapter | none | cpu: jax_region, gpu: jax_region | bounded | pass (`20260729T005942Z-5ade9aee`) |
| `examples/1_Simple/tracing_fieldlines_NCSX.py` | `examples/jax/1_Simple/tracing_fieldlines_NCSX.py` | eligible / adapter | none | cpu: jax_region, gpu: jax_region | bounded | pass (`20260729T005942Z-5ade9aee`) |
| `examples/1_Simple/tracing_fieldlines_QA.py` | `examples/jax/1_Simple/tracing_fieldlines_QA.py` | eligible / adapter | none | cpu: jax_region, gpu: jax_region | bounded | pass (`20260729T005942Z-5ade9aee`) |
| `examples/1_Simple/tracing_particle.py` | `examples/jax/1_Simple/tracing_particle.py` | eligible / adapter | none | cpu: jax_region, gpu: jax_region | bounded | pass (`20260729T005942Z-5ade9aee`) |
| `examples/2_Intermediate/B_external_normal.py` | — | blocked | VMEC | — | — | not run |
| `examples/2_Intermediate/QH_fixed_resolution.py` | — | blocked | VMEC | — | — | not run |
| `examples/2_Intermediate/QH_fixed_resolution_boozer.py` | — | blocked | VMEC | — | — | not run |
| `examples/2_Intermediate/QSC.py` | — | not_applicable | QSC | — | — | not run |
| `examples/2_Intermediate/boozer.py` | `examples/jax/2_Intermediate/boozer.py` | eligible / adapter | none | cpu: jax_region, gpu: jax_region | bounded | pass (`20260729T005942Z-5ade9aee`) |
| `examples/2_Intermediate/boozerQA.py` | `examples/jax/2_Intermediate/boozerQA.py` | eligible / adapter | none | cpu: jax_region, gpu: jax_region | bounded | pass (`20260729T005942Z-5ade9aee`) |
| `examples/2_Intermediate/boozerQA_ls_mpi.py` | — | blocked | MPI | — | — | not run |
| `examples/2_Intermediate/constrained_optimization.py` | — | blocked | VMEC | — | — | not run |
| `examples/2_Intermediate/eliminate_magnetic_islands.py` | — | blocked | SPEC | — | — | not run |
| `examples/2_Intermediate/free_boundary_vmec.py` | — | blocked | VMEC | — | — | not run |
| `examples/2_Intermediate/permanent_magnet_MUSE.py` | `examples/jax/2_Intermediate/permanent_magnet_MUSE.py` | eligible / mirror | VMEC (optional post-check, disabled by default) | cpu: full_workflow, gpu: full_workflow | bounded | pass (`20260729T005942Z-5ade9aee`) |
| `examples/2_Intermediate/permanent_magnet_PM4Stell.py` | `examples/jax/2_Intermediate/permanent_magnet_PM4Stell.py` | eligible / mirror | none | cpu: full_workflow, gpu: full_workflow | bounded | pass (`20260729T005942Z-5ade9aee`) |
| `examples/2_Intermediate/permanent_magnet_QA.py` | `examples/jax/2_Intermediate/permanent_magnet_QA.py` | eligible / mirror | VMEC (optional post-check, disabled by default) | cpu: full_workflow, gpu: full_workflow | bounded | pass (`20260729T005942Z-5ade9aee`) |
| `examples/2_Intermediate/resolution_increase.py` | — | blocked | VMEC | — | — | not run |
| `examples/2_Intermediate/resolution_increase_boozer.py` | — | blocked | VMEC | — | — | not run |
| `examples/2_Intermediate/stage_two_optimization.py` | `examples/jax/2_Intermediate/stage_two_optimization.py` | eligible / adapter | none | cpu: jax_region, gpu: jax_region | bounded | pass (`20260729T005942Z-5ade9aee`) |
| `examples/2_Intermediate/stage_two_optimization_finite_beta.py` | — | blocked | VMEC | — | — | not run |
| `examples/2_Intermediate/stage_two_optimization_planar_coils.py` | `examples/jax/2_Intermediate/stage_two_optimization_planar_coils.py` | eligible / adapter | none | cpu: jax_region, gpu: jax_region | bounded | pass (`20260729T005942Z-5ade9aee`) |
| `examples/2_Intermediate/stage_two_optimization_stochastic.py` | `examples/jax/2_Intermediate/stage_two_optimization_stochastic.py` | eligible / adapter | none | cpu: jax_region, gpu: jax_region | bounded | pass (`20260729T005942Z-5ade9aee`) |
| `examples/2_Intermediate/strain_optimization.py` | `examples/jax/2_Intermediate/strain_optimization.py` | eligible / adapter | none | cpu: jax_region, gpu: jax_region | bounded | pass (`20260729T005942Z-5ade9aee`) |
| `examples/2_Intermediate/tracing_boozer.py` | — | blocked | VMEC | — | — | not run |
| `examples/2_Intermediate/vmec_adjoint.py` | — | blocked | VMEC | — | — | not run |
| `examples/2_Intermediate/wireframe_gsco_modular.py` | `examples/jax/2_Intermediate/wireframe_gsco_modular.py` | eligible / adapter | none | cpu: jax_region, gpu: jax_region | bounded | pass (`20260729T005942Z-5ade9aee`) |
| `examples/2_Intermediate/wireframe_gsco_sector_saddle.py` | `examples/jax/2_Intermediate/wireframe_gsco_sector_saddle.py` | eligible / adapter | none | cpu: jax_region, gpu: jax_region | bounded | pass (`20260729T005942Z-5ade9aee`) |
| `examples/2_Intermediate/wireframe_rcls_basic.py` | `examples/jax/2_Intermediate/wireframe_rcls_basic.py` | eligible / adapter | none | cpu: jax_region, gpu: jax_region | bounded | pass (`20260729T005942Z-5ade9aee`) |
| `examples/2_Intermediate/wireframe_rcls_with_ports.py` | `examples/jax/2_Intermediate/wireframe_rcls_with_ports.py` | eligible / adapter | none | cpu: jax_region, gpu: jax_region | bounded | pass (`20260729T005942Z-5ade9aee`) |
| `examples/3_Advanced/coil_forces.py` | `examples/jax/3_Advanced/coil_forces.py` | eligible / adapter | none | cpu: jax_region, gpu: jax_region | bounded | pass (`20260729T005942Z-5ade9aee`) |
| `examples/3_Advanced/optimize_qs_and_islands_simultaneously.py` | — | blocked | SPEC, VMEC | — | — | not run |
| `examples/3_Advanced/single_stage_boozer_vacuum_optimization.py` | `examples/jax/3_Advanced/single_stage_boozer_vacuum_optimization.py` | eligible / adapter | none | cpu: jax_region, gpu: jax_region | bounded | pass (`20260729T005942Z-5ade9aee`) |
| `examples/3_Advanced/single_stage_optimization.py` | `examples/jax/3_Advanced/single_stage_optimization.py` | hybrid / hybrid | VMEC | cpu: host_and_jax_slice, gpu: jax_slice_only | not_applicable | unsupported |
| `examples/3_Advanced/single_stage_optimization_finite_beta.py` | — | blocked | VMEC | — | — | not run |
| `examples/3_Advanced/stage_two_optimization_finitebuild.py` | `examples/jax/3_Advanced/stage_two_optimization_finitebuild.py` | eligible / adapter | none | cpu: jax_region, gpu: jax_region | bounded | pass (`20260729T005942Z-5ade9aee`) |
| `examples/3_Advanced/wireframe_gsco_multistep.py` | `examples/jax/3_Advanced/wireframe_gsco_multistep.py` | eligible / adapter | none | cpu: jax_region, gpu: jax_region | bounded | pass (`20260729T005942Z-5ade9aee`) |
| `examples/stellarator_benchmarks/1DOF_circularCrossSection_varyAxis_targetIota.py` | — | blocked | VMEC | — | — | not run |
| `examples/stellarator_benchmarks/1DOF_circularCrossSection_varyAxis_targetIota_spec.py` | — | blocked | SPEC | — | — | not run |
| `examples/stellarator_benchmarks/1DOF_circularCrossSection_varyR0_targetVolume.py` | — | blocked | VMEC | — | — | not run |
| `examples/stellarator_benchmarks/1DOF_circularCrossSection_varyR0_targetVolume_spec.py` | — | blocked | SPEC | — | — | not run |
| `examples/stellarator_benchmarks/2DOF_circularCrossSection_varyAxis_targetIotaAndQuasisymmetry.py` | — | blocked | VMEC | — | — | not run |
| `examples/stellarator_benchmarks/2DOF_specOnly_targetIotaAndVolume.py` | — | blocked | SPEC | — | — | not run |
| `examples/stellarator_benchmarks/2DOF_vmecAndSpec.py` | — | blocked | SPEC, VMEC | — | — | not run |
| `examples/stellarator_benchmarks/2DOF_vmecOnly_targetIotaAndVolume.py` | — | blocked | VMEC | — | — | not run |
| `examples/stellarator_benchmarks/7dof.py` | — | blocked | VMEC | — | — | not run |

Regenerate with:

```bash
python -m examples.jax.native_to_jax_index --write
```

Verify it is current with `--check`.
