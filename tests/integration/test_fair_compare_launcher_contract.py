"""Contract for the same-node fair CPU-vs-GPU comparison launcher.

``benchmarks/perlmutter/single_stage_fair_compare_gpu.slurm`` exists to produce a
*defensible* cpu-vs-gpu comparison: one build on a single GPU node, then the
identical fullgraph mpol10 warm-started case run under ``--platform cpu`` and
``--platform cuda`` back-to-back, so only the device differs. These tests pin the
properties that make the comparison fair (and that a regression would silently
break):

- both device lanes actually run (not just one);
- the fullgraph lane forces ``--boozer-optimizer-backend ondevice`` (without it the
  CPU lane crashes at boozer_surface.py:5659 under jax_cpu_parity);
- host threads are capped (the ~50x oversubscription artifact);
- the mpol10 warm-start is required (cold high-res is contract-blocked);
- compile/cache-miss diagnostics are recorded by default
  (``--record-jax-compile-diagnostics``, now wired into the harness), gated by
  ``FAIR_RECORD_COMPILE_DIAGNOSTICS`` and applied identically on both lanes so the
  compile-vs-steady-state split and recompile counts stay a fair comparison.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "benchmarks" / "perlmutter" / "single_stage_fair_compare_gpu.slurm"


def _script() -> str:
    return LAUNCHER.read_text(encoding="utf-8")


def test_runs_both_device_lanes_on_one_node():
    script = _script()
    # Premise: it drives the parity harness, and both lanes execute.
    assert "benchmarks/single_stage_init_parity.py" in script
    assert re.search(r"^run_lane cpu$", script, re.MULTILINE), "cpu lane not invoked"
    assert re.search(r"^run_lane cuda$", script, re.MULTILINE), "cuda lane not invoked"
    # Same node: a single GPU allocation, not two separate jobs.
    assert "#SBATCH -C gpu" in script
    assert "--platform" in script  # parameterized per lane


def test_fullgraph_lane_forces_ondevice_inner_boozer():
    script = _script()
    assert "--optimizer-backend scipy-jax-fullgraph" in script
    assert "--boozer-optimizer-backend ondevice" in script
    # The dropped on-device outer lane must not reappear.
    assert "--optimizer-backend ondevice" not in script


def test_threads_capped_and_cpu_lane_hides_gpu():
    script = _script()
    assert "OMP_NUM_THREADS" in script
    assert "OPENBLAS_NUM_THREADS" in script
    # The CPU lane must not see the GPU, or its "cpu" timing is not a CPU timing.
    assert 'CUDA_VISIBLE_DEVICES=""' in script


def test_mpol10_warm_start_is_required():
    script = _script()
    # Fail-closed required var (cold high-res single-stage is contract-blocked).
    assert re.search(r"FAIR_WARM_START_RUN_DIR:\?", script), (
        "launcher does not require FAIR_WARM_START_RUN_DIR"
    )
    assert "--warm-start-run-dir" in script


def test_records_compile_diagnostics_by_default_identically_on_both_lanes():
    # single_stage_init_parity.py now exposes --record-jax-compile-diagnostics and
    # threads it to the JAX target-lane child, so the fair launcher records
    # compile/cache-miss diagnostics by default to separate one-time compile cost
    # from steady-state throughput and to surface GPU XLA recompile counts.
    script = _script()
    assert "--record-jax-compile-diagnostics" in script
    # Default-on, but escapable for a pristine throughput run.
    assert "FAIR_RECORD_COMPILE_DIAGNOSTICS:-1" in script
    # Built once in the shared run_lane invocation -> identical on cpu and cuda
    # (a per-lane divergence here would make the comparison unfair).
    assert "compile_diag_args=(--record-jax-compile-diagnostics)" in script
    assert '"${compile_diag_args[@]}"' in script


def test_launcher_records_mixed_parity_timing_label():
    script = _script()
    assert "FAIR_TIMING_CLASSIFICATION" in script
    assert "mixed-parity-reference" in script
    assert "FAIR_SUPPORTS_PERFORMANCE_HEADLINE" in script
    assert "benchmark_timing_label.json" in script
