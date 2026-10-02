# Native-to-JAX example index

Generated from `manifest.json`. Do not edit this table by hand.
Official upstream scope: `https://github.com/hiddenSymmetries/simsopt` `master` at `9e027eac38028d57aa23777be52a781aa860e347` (53 source files).

## Official upstream catalog

| Native example | JAX mirror | Classification | Runtime dependencies | Device scope |
| --- | --- | --- | --- | --- |
| `examples/1_Simple/just_a_quadratic.py` | `examples/jax/1_Simple/just_a_quadratic.py` | eligible / adapter | none | outer: CPU SciPy; cpu: jax_region, gpu: jax_region |
| `examples/1_Simple/logger_example.py` | — | not_applicable | none | — |
| `examples/1_Simple/minimize_curve_length.py` | `examples/jax/1_Simple/minimize_curve_length.py` | eligible / adapter | none | outer: CPU SciPy; cpu: jax_region, gpu: jax_region |
| `examples/1_Simple/periodicfieldline_QA.py` | — | blocked | none | — |
| `examples/1_Simple/periodicfieldline_QH.py` | — | blocked | none | — |
| `examples/1_Simple/permanent_magnet_simple.py` | `examples/jax/1_Simple/permanent_magnet_simple.py` | eligible / mirror | none | cpu: full_workflow, gpu: full_workflow |
| `examples/1_Simple/qfm.py` | `examples/jax/1_Simple/qfm.py` | eligible / adapter | none | outer: CPU SciPy; cpu: jax_region, gpu: jax_region |
| `examples/1_Simple/stage_two_optimization_minimal.py` | `examples/jax/1_Simple/stage_two_optimization_minimal.py` | eligible / adapter | none | outer: CPU SciPy; cpu: jax_region, gpu: jax_region |
| `examples/1_Simple/surf_vol_area.py` | `examples/jax/1_Simple/surf_vol_area.py` | eligible / adapter | none | outer: CPU SciPy; cpu: jax_region, gpu: jax_region |
| `examples/1_Simple/tracing_fieldlines_NCSX.py` | `examples/jax/1_Simple/tracing_fieldlines_NCSX.py` | eligible / adapter | none | cpu: jax_region, gpu: jax_region |
| `examples/1_Simple/tracing_fieldlines_QA.py` | `examples/jax/1_Simple/tracing_fieldlines_QA.py` | eligible / adapter | none | cpu: jax_region, gpu: jax_region |
| `examples/1_Simple/tracing_particle.py` | `examples/jax/1_Simple/tracing_particle.py` | eligible / adapter | none | cpu: jax_region, gpu: jax_region |
| `examples/2_Intermediate/B_external_normal.py` | — | blocked | VMEC | — |
| `examples/2_Intermediate/QH_fixed_resolution.py` | — | blocked | VMEC | — |
| `examples/2_Intermediate/QH_fixed_resolution_boozer.py` | — | blocked | VMEC | — |
| `examples/2_Intermediate/QSC.py` | — | not_applicable | QSC | — |
| `examples/2_Intermediate/boozer.py` | `examples/jax/2_Intermediate/boozer.py` | eligible / adapter | none | cpu: jax_region, gpu: jax_region |
| `examples/2_Intermediate/boozerQA.py` | `examples/jax/2_Intermediate/boozerQA.py` | eligible / adapter | none | cpu: jax_region, gpu: jax_region |
| `examples/2_Intermediate/boozerQA_ls_mpi.py` | — | blocked | MPI | — |
| `examples/2_Intermediate/constrained_optimization.py` | — | blocked | VMEC | — |
| `examples/2_Intermediate/eliminate_magnetic_islands.py` | — | blocked | SPEC | — |
| `examples/2_Intermediate/free_boundary_vmec.py` | — | blocked | VMEC | — |
| `examples/2_Intermediate/permanent_magnet_MUSE.py` | `examples/jax/2_Intermediate/permanent_magnet_MUSE.py` | eligible / mirror | VMEC (optional post-check, disabled by default) | cpu: full_workflow, gpu: full_workflow |
| `examples/2_Intermediate/permanent_magnet_PM4Stell.py` | `examples/jax/2_Intermediate/permanent_magnet_PM4Stell.py` | eligible / mirror | none | cpu: full_workflow, gpu: full_workflow |
| `examples/2_Intermediate/permanent_magnet_QA.py` | `examples/jax/2_Intermediate/permanent_magnet_QA.py` | eligible / mirror | VMEC (optional post-check, disabled by default) | cpu: full_workflow, gpu: full_workflow |
| `examples/2_Intermediate/resolution_increase.py` | — | blocked | VMEC | — |
| `examples/2_Intermediate/resolution_increase_boozer.py` | — | blocked | VMEC | — |
| `examples/2_Intermediate/stage_two_optimization.py` | `examples/jax/2_Intermediate/stage_two_optimization.py` | eligible / adapter | none | outer: CPU SciPy; cpu: jax_region, gpu: jax_region |
| `examples/2_Intermediate/stage_two_optimization_finite_beta.py` | — | blocked | VMEC | — |
| `examples/2_Intermediate/stage_two_optimization_planar_coils.py` | `examples/jax/2_Intermediate/stage_two_optimization_planar_coils.py` | eligible / adapter | none | outer: CPU SciPy; cpu: jax_region, gpu: jax_region |
| `examples/2_Intermediate/stage_two_optimization_stochastic.py` | `examples/jax/2_Intermediate/stage_two_optimization_stochastic.py` | eligible / adapter | none | outer: CPU SciPy; cpu: jax_region, gpu: jax_region |
| `examples/2_Intermediate/strain_optimization.py` | `examples/jax/2_Intermediate/strain_optimization.py` | eligible / adapter | none | cpu: jax_region, gpu: jax_region |
| `examples/2_Intermediate/tracing_boozer.py` | — | blocked | VMEC | — |
| `examples/2_Intermediate/vmec_adjoint.py` | — | blocked | VMEC | — |
| `examples/2_Intermediate/wireframe_gsco_modular.py` | `examples/jax/2_Intermediate/wireframe_gsco_modular.py` | eligible / adapter | none | cpu: jax_region, gpu: jax_region |
| `examples/2_Intermediate/wireframe_gsco_sector_saddle.py` | `examples/jax/2_Intermediate/wireframe_gsco_sector_saddle.py` | eligible / adapter | none | cpu: jax_region, gpu: jax_region |
| `examples/2_Intermediate/wireframe_rcls_basic.py` | `examples/jax/2_Intermediate/wireframe_rcls_basic.py` | eligible / adapter | none | cpu: jax_region, gpu: jax_region |
| `examples/2_Intermediate/wireframe_rcls_with_ports.py` | `examples/jax/2_Intermediate/wireframe_rcls_with_ports.py` | eligible / adapter | none | cpu: jax_region, gpu: jax_region |
| `examples/3_Advanced/coil_forces.py` | `examples/jax/3_Advanced/coil_forces.py` | eligible / adapter | none | outer: CPU SciPy; cpu: jax_region, gpu: jax_region |
| `examples/3_Advanced/optimize_qs_and_islands_simultaneously.py` | — | blocked | SPEC, VMEC | — |
| `examples/3_Advanced/single_stage_optimization.py` | `examples/jax/3_Advanced/single_stage_optimization.py` | hybrid / hybrid | VMEC | cpu: host_and_jax_slice, gpu: jax_slice_only |
| `examples/3_Advanced/single_stage_optimization_finite_beta.py` | — | blocked | VMEC | — |
| `examples/3_Advanced/stage_two_optimization_finitebuild.py` | `examples/jax/3_Advanced/stage_two_optimization_finitebuild.py` | eligible / adapter | none | outer: CPU SciPy; cpu: jax_region, gpu: jax_region |
| `examples/3_Advanced/wireframe_gsco_multistep.py` | `examples/jax/3_Advanced/wireframe_gsco_multistep.py` | eligible / adapter | none | cpu: jax_region, gpu: jax_region |
| `examples/stellarator_benchmarks/1DOF_circularCrossSection_varyAxis_targetIota.py` | — | blocked | VMEC | — |
| `examples/stellarator_benchmarks/1DOF_circularCrossSection_varyAxis_targetIota_spec.py` | — | blocked | SPEC | — |
| `examples/stellarator_benchmarks/1DOF_circularCrossSection_varyR0_targetVolume.py` | — | blocked | VMEC | — |
| `examples/stellarator_benchmarks/1DOF_circularCrossSection_varyR0_targetVolume_spec.py` | — | blocked | SPEC | — |
| `examples/stellarator_benchmarks/2DOF_circularCrossSection_varyAxis_targetIotaAndQuasisymmetry.py` | — | blocked | VMEC | — |
| `examples/stellarator_benchmarks/2DOF_specOnly_targetIotaAndVolume.py` | — | blocked | SPEC | — |
| `examples/stellarator_benchmarks/2DOF_vmecAndSpec.py` | — | blocked | SPEC, VMEC | — |
| `examples/stellarator_benchmarks/2DOF_vmecOnly_targetIotaAndVolume.py` | — | blocked | VMEC | — |
| `examples/stellarator_benchmarks/7dof.py` | — | blocked | VMEC | — |

Regenerate with:

```bash
python -m examples.jax.native_to_jax_index --write
```

`--check` verifies index consistency.
