"""Pinned upstream authority for the official native example inventory.

This module is intentionally offline: validating the active manifests must not
depend on a local clone of the upstream remote or on network access.
"""

from __future__ import annotations

from typing import Final

OFFICIAL_UPSTREAM_REPOSITORY: Final = "https://github.com/hiddenSymmetries/simsopt"
OFFICIAL_UPSTREAM_DEFAULT_BRANCH: Final = "master"
OFFICIAL_UPSTREAM_COMMIT: Final = "9e027eac38028d57aa23777be52a781aa860e347"

# Paths are relative to the upstream ``examples`` directory at the pinned
# commit above. Their order is the canonical official inventory order.
OFFICIAL_NATIVE_EXAMPLE_SOURCES: Final[tuple[str, ...]] = (
    "1_Simple/just_a_quadratic.py",
    "1_Simple/logger_example.py",
    "1_Simple/minimize_curve_length.py",
    "1_Simple/periodicfieldline_QA.py",
    "1_Simple/periodicfieldline_QH.py",
    "1_Simple/permanent_magnet_simple.py",
    "1_Simple/qfm.py",
    "1_Simple/stage_two_optimization_minimal.py",
    "1_Simple/surf_vol_area.py",
    "1_Simple/tracing_fieldlines_NCSX.py",
    "1_Simple/tracing_fieldlines_QA.py",
    "1_Simple/tracing_particle.py",
    "2_Intermediate/B_external_normal.py",
    "2_Intermediate/QH_fixed_resolution.py",
    "2_Intermediate/QH_fixed_resolution_boozer.py",
    "2_Intermediate/QSC.py",
    "2_Intermediate/boozer.py",
    "2_Intermediate/boozerQA.py",
    "2_Intermediate/boozerQA_ls_mpi.py",
    "2_Intermediate/constrained_optimization.py",
    "2_Intermediate/eliminate_magnetic_islands.py",
    "2_Intermediate/free_boundary_vmec.py",
    "2_Intermediate/permanent_magnet_MUSE.py",
    "2_Intermediate/permanent_magnet_PM4Stell.py",
    "2_Intermediate/permanent_magnet_QA.py",
    "2_Intermediate/resolution_increase.py",
    "2_Intermediate/resolution_increase_boozer.py",
    "2_Intermediate/stage_two_optimization.py",
    "2_Intermediate/stage_two_optimization_finite_beta.py",
    "2_Intermediate/stage_two_optimization_planar_coils.py",
    "2_Intermediate/stage_two_optimization_stochastic.py",
    "2_Intermediate/strain_optimization.py",
    "2_Intermediate/tracing_boozer.py",
    "2_Intermediate/vmec_adjoint.py",
    "2_Intermediate/wireframe_gsco_modular.py",
    "2_Intermediate/wireframe_gsco_sector_saddle.py",
    "2_Intermediate/wireframe_rcls_basic.py",
    "2_Intermediate/wireframe_rcls_with_ports.py",
    "3_Advanced/coil_forces.py",
    "3_Advanced/optimize_qs_and_islands_simultaneously.py",
    "3_Advanced/single_stage_optimization.py",
    "3_Advanced/single_stage_optimization_finite_beta.py",
    "3_Advanced/stage_two_optimization_finitebuild.py",
    "3_Advanced/wireframe_gsco_multistep.py",
    "stellarator_benchmarks/1DOF_circularCrossSection_varyAxis_targetIota.py",
    "stellarator_benchmarks/1DOF_circularCrossSection_varyAxis_targetIota_spec.py",
    "stellarator_benchmarks/1DOF_circularCrossSection_varyR0_targetVolume.py",
    "stellarator_benchmarks/1DOF_circularCrossSection_varyR0_targetVolume_spec.py",
    "stellarator_benchmarks/2DOF_circularCrossSection_varyAxis_targetIotaAndQuasisymmetry.py",
    "stellarator_benchmarks/2DOF_specOnly_targetIotaAndVolume.py",
    "stellarator_benchmarks/2DOF_vmecAndSpec.py",
    "stellarator_benchmarks/2DOF_vmecOnly_targetIotaAndVolume.py",
    "stellarator_benchmarks/7dof.py",
)
