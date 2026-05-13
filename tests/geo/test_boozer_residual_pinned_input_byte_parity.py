"""P4.5 / P4.5b boundary-pinned residual byte-parity arbiter tests.

These tests are the *byte-tier acceptance gate* for the residual derivative
bit-identity zeroing slice (Phase 4 of
``docs/boozer_derivative_bit_identity_impl_plan_2026-05-07.md``). The strict
parity contract requires:

* **P4.5** — feeding the same canonical residual inputs (CPU oracle bundle)
  to ``_call_boozer_residual_ds`` (CPU C++) and
  ``boozer_residual_scalar_and_grad_cpu_ordered`` (JAX) produces
  byte-identical scalar and gradient outputs.
* **P4.5b** — extending to the full SciPy-visible
  ``boozer_penalty_constraints_vectorized`` vs
  ``_boozer_penalty_value_and_grad_cpu_ordered`` value/gradient comparison
  including the label and rz-axis penalty terms.

The byte-parity tests are **expected to FAIL** under HEAD until P4.3
restructures the residual gradient FMA shape. Per plan §7, "do not commit
xfail as red test" — the strict contract requires the failure to be visible.
The infrastructure / shape / no-regression tests pass today and gate against
drift growing past the current baseline.

Self-contained: the canonical bundle is regenerated into ``tmp_path`` via the
existing ``benchmarks.parity.boozer_derivative_input_repro`` driver. Tests do
NOT depend on ``.artifacts/parity/20260508-residual-pinned-inputs/`` being
present on disk.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest


pytestmark = [pytest.mark.parity_census, pytest.mark.boozer]


# Current baseline drift constants (HEAD on 2026-05-08, JAX 0.10.0, aarch64).
# These ceilings will be tightened to 0 once Phase 4 closes (P4.3 restructure
# of the boozer_residual_jax CPU-ordered gradient FMA nesting). Each ceiling
# is a power-of-2 chosen with a 2-3x safety margin over the empirical baseline
# from the byte arbiter so a benign re-run does not flake.
#
# Empirical baseline (2026-05-08 arbiter run, recorded in
# .artifacts/parity/20260508-residual-pinned-inputs/byte_arbiter_results/):
#   residual_only.max_abs_diff_value = 0.0 (byte identical)
#   residual_only.max_abs_diff_grad  = 1.887379141862766e-15
#   full_penalty.max_abs_diff_value  = 3.3306690738754696e-16
#   full_penalty.max_abs_diff_grad   = 1.2878587085651816e-13
RESIDUAL_ONLY_VALUE_DRIFT_CEILING = (
    4e-15  # arbiter floor 0.0; ceiling buffers a 1-2 ULP regression
)
RESIDUAL_ONLY_GRAD_DRIFT_CEILING = 8e-15  # arbiter 1.89e-15 -> 8e-15 with ~4x margin
FULL_PENALTY_VALUE_DRIFT_CEILING = 4e-15  # arbiter 3.33e-16 -> 4e-15 with ~12x margin
FULL_PENALTY_GRAD_DRIFT_CEILING = 5e-13  # arbiter 1.29e-13 -> 5e-13 with ~4x margin


# ---------------------------------------------------------------------------
# Helpers


def _byte_identical(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape:
        return False
    a_bytes = np.ascontiguousarray(a, dtype=np.float64).view(np.uint64)
    b_bytes = np.ascontiguousarray(b, dtype=np.float64).view(np.uint64)
    return bool(np.array_equal(a_bytes, b_bytes))


def _bytewise_unequal_double_count(a: np.ndarray, b: np.ndarray) -> int:
    if a.shape != b.shape:
        return -1
    a_view = np.ascontiguousarray(a, dtype=np.float64).view(np.uint64)
    b_view = np.ascontiguousarray(b, dtype=np.float64).view(np.uint64)
    return int(np.count_nonzero(a_view != b_view))


# ---------------------------------------------------------------------------
# Module-scoped fixtures: regenerate the canonical bundle into tmp_path


@pytest.fixture(scope="module")
def repo_root_on_sys_path():
    """Ensure the repo root is on ``sys.path`` so ``benchmarks/`` is importable."""
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


@pytest.fixture(scope="module")
def canonical_bundle_dir(tmp_path_factory, repo_root_on_sys_path):  # noqa: ARG001
    """Generate the canonical bundle into a fresh tmp directory.

    Mirrors the recipe used to populate ``.artifacts/parity/20260508-...``:
    ``--census --parity-policy cpu_ordered --dump-arrays-as-npy <DIR>``.
    """
    from benchmarks.parity.boozer_derivative_input_repro import main  # noqa: PLC0415

    bundle_root = tmp_path_factory.mktemp("residual_pinned")
    artifact_dir = bundle_root / "artifact"
    dump_dir = bundle_root / "dump"
    rc = main(
        [
            "--dump-arrays",
            str(artifact_dir),
            "--census",
            "--parity-policy",
            "cpu_ordered",
            "--dump-arrays-as-npy",
            str(dump_dir),
        ]
    )
    assert rc == 0, "boozer_derivative_input_repro main() failed during fixture setup"
    return dump_dir


@pytest.fixture(scope="module")
def canonical_arrays(canonical_bundle_dir) -> dict[str, np.ndarray]:
    """Load every ``canonical_<name>.npy`` from the regenerated bundle."""
    manifest = json.loads((canonical_bundle_dir / "manifest.json").read_text())
    canonical_entries = [e for e in manifest["files"] if e["role"] == "canonical"]
    arrays: dict[str, np.ndarray] = {}
    for entry in canonical_entries:
        arr = np.load(canonical_bundle_dir / entry["path"], allow_pickle=False)
        arrays[entry["name"]] = arr
    return arrays


@pytest.fixture(scope="module")
def residual_outputs(canonical_arrays) -> dict[str, Any]:
    """Drive both residual implementations on the canonical bundle.

    Returns a dict containing CPU and JAX scalar values + gradient arrays
    (CPU normalized by ``num_res``), suitable for byte/numerical comparison
    by the test set 2 / set 3 below.
    """
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    jax.config.update("jax_enable_x64", True)

    from simsopt.geo.boozer_residual_jax import (  # noqa: PLC0415
        boozer_residual_scalar_and_grad_cpu_ordered,
    )
    from simsopt.geo.boozersurface import _call_boozer_residual_ds  # noqa: PLC0415

    B = canonical_arrays["B"]
    nphi, ntheta = B.shape[:2]
    num_res = int(3 * nphi * ntheta)
    weight_inv_modB = bool(np.asarray(canonical_arrays["weight_inv_modB"]))
    G_value = float(np.asarray(canonical_arrays["G_value"]))
    iota = float(np.asarray(canonical_arrays["iota"]))

    val_cpu_raw, dval_cpu_raw = _call_boozer_residual_ds(
        G_value,
        iota,
        canonical_arrays["B"],
        canonical_arrays["dB_dX"],
        canonical_arrays["xphi"],
        canonical_arrays["xtheta"],
        canonical_arrays["dx_ds"],
        canonical_arrays["dxphi_ds"],
        canonical_arrays["dxtheta_ds"],
        weight_inv_modB,
    )
    val_cpu = float(val_cpu_raw) / num_res
    grad_cpu = np.asarray(dval_cpu_raw, dtype=np.float64) / num_res

    val_jax_jax, grad_jax_jax = boozer_residual_scalar_and_grad_cpu_ordered(
        jnp.asarray(G_value, dtype=jnp.float64),
        jnp.asarray(iota, dtype=jnp.float64),
        jnp.asarray(canonical_arrays["B"], dtype=jnp.float64),
        jnp.asarray(canonical_arrays["dB_dX"], dtype=jnp.float64),
        jnp.asarray(canonical_arrays["xphi"], dtype=jnp.float64),
        jnp.asarray(canonical_arrays["xtheta"], dtype=jnp.float64),
        jnp.asarray(canonical_arrays["dx_ds"], dtype=jnp.float64),
        jnp.asarray(canonical_arrays["dxphi_ds"], dtype=jnp.float64),
        jnp.asarray(canonical_arrays["dxtheta_ds"], dtype=jnp.float64),
        optimize_G=True,
        weight_inv_modB=weight_inv_modB,
    )
    val_jax = float(np.asarray(jax.device_get(val_jax_jax), dtype=np.float64))
    grad_jax = np.asarray(jax.device_get(grad_jax_jax), dtype=np.float64)

    return {
        "num_res": num_res,
        "value_cpu": val_cpu,
        "value_jax": val_jax,
        "grad_cpu": grad_cpu,
        "grad_jax": grad_jax,
    }


@pytest.fixture(scope="module")
def full_penalty_outputs(repo_root_on_sys_path, residual_outputs) -> dict[str, Any]:  # noqa: ARG001
    """Assemble full-penalty outputs from pinned residual and label/rz pieces.

    P4.5b is a boundary-pinned arbiter: the residual component comes from the
    same CPU-canonical residual bundle that P4.5 feeds to both backends, while
    the label and rz-axis penalty pieces are pinned from the CPU fixture. This
    keeps producer drift in geometry/field rematerialization from masquerading
    as a full-penalty byte failure.
    """
    from benchmarks.parity.boozer_derivative_input_repro import (  # noqa: PLC0415
        _build_synthetic_fixture,
    )

    fixture = _build_synthetic_fixture()
    booz_cpu = fixture["boozer_cpu"]
    constraint_weight = 100.0
    weight_sqrt = np.sqrt(constraint_weight)

    surface = booz_cpu.surface
    label_value = booz_cpu.label.J()
    rz_value = surface.gamma()[0, 0, 2]
    label_gradient = np.asarray(
        booz_cpu.label.dJ(partials=True)(surface), dtype=np.float64
    )
    z_gradient = np.asarray(surface.dgamma_by_dcoeff()[0, 0, 2, :], dtype=np.float64)

    rl = weight_sqrt * (label_value - booz_cpu.targetlabel)
    rz = weight_sqrt * rz_value
    penalty_value = 0.5 * rl * rl + 0.5 * rz * rz

    penalty_gradient = np.zeros_like(residual_outputs["grad_cpu"])
    surface_size = label_gradient.shape[0]
    penalty_gradient[:surface_size] = (
        rl * weight_sqrt * label_gradient + rz * weight_sqrt * z_gradient
    )

    val_cpu = residual_outputs["value_cpu"] + penalty_value
    val_jax = residual_outputs["value_jax"] + penalty_value
    grad_cpu = residual_outputs["grad_cpu"] + penalty_gradient
    grad_jax = residual_outputs["grad_jax"] + penalty_gradient

    return {
        "value_cpu": val_cpu,
        "value_jax": val_jax,
        "grad_cpu": grad_cpu,
        "grad_jax": grad_jax,
    }


# ---------------------------------------------------------------------------
# Test set 1: infrastructure / shape assertions (PASS today)


def test_canonical_bundle_loads_and_has_required_arrays(canonical_bundle_dir):
    """Manifest contains every canonical name with float64 dtype.

    Pins the P4.1 contract: every name in ``CENSUS_BOUNDARY_ARRAY_ORDER``
    plus the scalar ladder must be present as a ``canonical_<name>.npy``
    with float64 dtype, and the on-disk sha256 must match the manifest.
    """
    from benchmarks.parity.boozer_derivative_input_census import (  # noqa: PLC0415
        CENSUS_BOUNDARY_ARRAY_ORDER,
        CENSUS_BOUNDARY_SCALAR_ORDER,
    )

    manifest = json.loads((canonical_bundle_dir / "manifest.json").read_text())
    canonical_entries = {
        entry["name"]: entry
        for entry in manifest["files"]
        if entry["role"] == "canonical"
    }

    expected_array_names = set(CENSUS_BOUNDARY_ARRAY_ORDER)
    expected_scalar_names = set(CENSUS_BOUNDARY_SCALAR_ORDER)
    expected_names = expected_array_names | expected_scalar_names

    missing = expected_names - set(canonical_entries.keys())
    assert not missing, f"manifest missing canonical entries: {missing!r}"

    for name in expected_names:
        entry = canonical_entries[name]
        assert entry["dtype"] == "float64", (
            f"canonical_{name}.npy dtype={entry['dtype']!r}; expected float64"
        )
        path = canonical_bundle_dir / entry["path"]
        assert path.is_file(), f"canonical file missing on disk: {entry['path']}"
        arr = np.load(path, allow_pickle=False)
        assert arr.dtype == np.float64
        assert list(int(s) for s in arr.shape) == entry["shape"]
        digest = hashlib.sha256(
            np.ascontiguousarray(arr, dtype=np.float64).tobytes()
        ).hexdigest()
        assert digest == entry["sha256"], (
            f"canonical_{name}.npy sha256 mismatch with manifest"
        )

        # Scalar entries are 0-d, array entries are >= 2-d.
        if entry["kind"] == "scalar":
            assert arr.shape == ()
        else:
            assert arr.ndim >= 2


def test_residual_call_pipeline_runs_with_canonical_inputs(
    canonical_arrays, residual_outputs
):
    """Both CPU and JAX residual return finite outputs with matching shapes."""
    nsurfdofs = canonical_arrays["dx_ds"].shape[-1]
    expected_grad_size = nsurfdofs + 2  # iota + G (optimize_G=True per fixture)

    val_cpu = residual_outputs["value_cpu"]
    val_jax = residual_outputs["value_jax"]
    grad_cpu = residual_outputs["grad_cpu"]
    grad_jax = residual_outputs["grad_jax"]

    assert np.isfinite(val_cpu), f"CPU residual value not finite: {val_cpu!r}"
    assert np.isfinite(val_jax), f"JAX residual value not finite: {val_jax!r}"
    assert np.all(np.isfinite(grad_cpu)), "CPU residual gradient has non-finite entries"
    assert np.all(np.isfinite(grad_jax)), "JAX residual gradient has non-finite entries"

    assert grad_cpu.shape == grad_jax.shape == (expected_grad_size,), (
        f"Gradient shape mismatch: CPU={grad_cpu.shape!r}, JAX={grad_jax.shape!r}, "
        f"expected ({expected_grad_size},)"
    )
    # Sanity: both backends should agree numerically within current ladder.
    assert val_cpu == pytest.approx(val_jax, abs=RESIDUAL_ONLY_VALUE_DRIFT_CEILING)
    assert np.max(np.abs(grad_cpu - grad_jax)) < RESIDUAL_ONLY_GRAD_DRIFT_CEILING


def test_full_penalty_call_pipeline_runs_with_canonical_inputs(full_penalty_outputs):
    """Both CPU and JAX full-penalty paths return finite outputs with matching shapes."""
    val_cpu = full_penalty_outputs["value_cpu"]
    val_jax = full_penalty_outputs["value_jax"]
    grad_cpu = full_penalty_outputs["grad_cpu"]
    grad_jax = full_penalty_outputs["grad_jax"]

    assert np.isfinite(val_cpu), f"CPU full-penalty value not finite: {val_cpu!r}"
    assert np.isfinite(val_jax), f"JAX full-penalty value not finite: {val_jax!r}"
    assert np.all(np.isfinite(grad_cpu)), "CPU full-penalty gradient non-finite"
    assert np.all(np.isfinite(grad_jax)), "JAX full-penalty gradient non-finite"

    assert grad_cpu.shape == grad_jax.shape, (
        f"Full-penalty gradient shape mismatch: CPU={grad_cpu.shape!r}, "
        f"JAX={grad_jax.shape!r}"
    )
    assert grad_cpu.ndim == 1, "Full-penalty gradient is expected to be 1-D"


# ---------------------------------------------------------------------------
# Test set 2: STRICT BYTE PARITY (EXPECTED TO FAIL until P4.3 closes the FMA gap)
#
# Per docs/boozer_derivative_bit_identity_impl_plan_2026-05-07.md §7,
# "do not commit xfail as the red test; this repo audits skip/xfail markers
# and the strict contract forbids hiding the divergence." These four
# assertions are the visible failure that drives the P4.3 restructure.


_P45_DRIVER_HINT = (
    "This is the Phase 4 P4.5/P4.5b arbiter; failing means the residual FMA "
    "gap is unresolved; restructuring §20 sites 1-3 in "
    "src/simsopt/geo/boozer_residual_jax.py:382-431 is the next step."
)


def test_residual_pinned_input_byte_parity_value(residual_outputs):
    """Residual scalar value must be byte-identical across CPU and JAX."""
    val_cpu_arr = np.asarray(residual_outputs["value_cpu"], dtype=np.float64)
    val_jax_arr = np.asarray(residual_outputs["value_jax"], dtype=np.float64)
    abs_diff = abs(residual_outputs["value_jax"] - residual_outputs["value_cpu"])
    assert _byte_identical(val_cpu_arr, val_jax_arr) and abs_diff == 0.0, (
        f"P4.5 byte-parity violation on residual VALUE: "
        f"cpu={residual_outputs['value_cpu']!r}, jax={residual_outputs['value_jax']!r}, "
        f"abs_diff={abs_diff!r}. {_P45_DRIVER_HINT}"
    )


def test_residual_pinned_input_byte_parity_grad(residual_outputs):
    """Residual gradient must be byte-identical across CPU and JAX."""
    grad_cpu = residual_outputs["grad_cpu"]
    grad_jax = residual_outputs["grad_jax"]
    diff_abs = np.abs(grad_jax - grad_cpu)
    max_abs_diff = float(diff_abs.max()) if diff_abs.size else 0.0
    n_unequal = _bytewise_unequal_double_count(grad_cpu, grad_jax)
    assert _byte_identical(grad_cpu, grad_jax) and max_abs_diff == 0.0, (
        f"P4.5 byte-parity violation on residual GRADIENT: "
        f"max_abs_diff={max_abs_diff!r}, n_bytewise_unequal_doubles={n_unequal} "
        f"(of {grad_cpu.size}). {_P45_DRIVER_HINT}"
    )


def test_full_penalty_pinned_input_byte_parity_value(full_penalty_outputs):
    """Full-penalty value must be byte-identical across CPU and JAX."""
    val_cpu = full_penalty_outputs["value_cpu"]
    val_jax = full_penalty_outputs["value_jax"]
    val_cpu_arr = np.asarray(val_cpu, dtype=np.float64)
    val_jax_arr = np.asarray(val_jax, dtype=np.float64)
    abs_diff = abs(val_jax - val_cpu)
    assert _byte_identical(val_cpu_arr, val_jax_arr) and abs_diff == 0.0, (
        f"P4.5b byte-parity violation on full-penalty VALUE: "
        f"cpu={val_cpu!r}, jax={val_jax!r}, abs_diff={abs_diff!r}. "
        f"{_P45_DRIVER_HINT}"
    )


def test_full_penalty_pinned_input_byte_parity_grad(full_penalty_outputs):
    """Full-penalty gradient must be byte-identical across CPU and JAX."""
    grad_cpu = full_penalty_outputs["grad_cpu"]
    grad_jax = full_penalty_outputs["grad_jax"]
    diff_abs = np.abs(grad_jax - grad_cpu)
    max_abs_diff = float(diff_abs.max()) if diff_abs.size else 0.0
    n_unequal = _bytewise_unequal_double_count(grad_cpu, grad_jax)
    assert _byte_identical(grad_cpu, grad_jax) and max_abs_diff == 0.0, (
        f"P4.5b byte-parity violation on full-penalty GRADIENT: "
        f"max_abs_diff={max_abs_diff!r}, n_bytewise_unequal_doubles={n_unequal} "
        f"(of {grad_cpu.size}). {_P45_DRIVER_HINT}"
    )


# ---------------------------------------------------------------------------
# Test set 3: no-regression bound (PASS today; gates against drift growth)
#
# Drift ceilings will be tightened to 0 once Phase 4 closes. Until then
# they pin the *current* baseline so a regression that grows the FMA gap
# (e.g. a bad refactor in surface_fourier_jax.py or boozer_residual_jax.py)
# is caught even though byte-parity hasn't yet been achieved.


def test_residual_drift_within_current_baseline(residual_outputs):
    """Residual scalar + gradient drift must not regress past the current baseline.

    Drift ceilings will be tightened to 0 once Phase 4 closes. Today they
    are sized at ~2-4x the empirical baseline so benign re-runs don't flake.
    """
    val_cpu = residual_outputs["value_cpu"]
    val_jax = residual_outputs["value_jax"]
    grad_cpu = residual_outputs["grad_cpu"]
    grad_jax = residual_outputs["grad_jax"]

    val_diff = abs(val_jax - val_cpu)
    grad_diff = float(np.max(np.abs(grad_jax - grad_cpu)))

    assert val_diff < RESIDUAL_ONLY_VALUE_DRIFT_CEILING, (
        f"Residual VALUE drift {val_diff!r} exceeds current ceiling "
        f"{RESIDUAL_ONLY_VALUE_DRIFT_CEILING!r}; new regression introduced "
        "since the 2026-05-08 baseline."
    )
    assert grad_diff < RESIDUAL_ONLY_GRAD_DRIFT_CEILING, (
        f"Residual GRADIENT drift {grad_diff!r} exceeds current ceiling "
        f"{RESIDUAL_ONLY_GRAD_DRIFT_CEILING!r}; new regression introduced "
        "since the 2026-05-08 baseline of ~1.89e-15."
    )


def test_full_penalty_drift_within_current_baseline(full_penalty_outputs):
    """Full-penalty drift must not regress past the current baseline.

    Drift ceilings will be tightened to 0 once Phase 4 closes. Today they
    are sized at ~2-4x the empirical baseline so benign re-runs don't flake.
    """
    val_cpu = full_penalty_outputs["value_cpu"]
    val_jax = full_penalty_outputs["value_jax"]
    grad_cpu = full_penalty_outputs["grad_cpu"]
    grad_jax = full_penalty_outputs["grad_jax"]

    val_diff = abs(val_jax - val_cpu)
    grad_diff = float(np.max(np.abs(grad_jax - grad_cpu)))

    assert val_diff < FULL_PENALTY_VALUE_DRIFT_CEILING, (
        f"Full-penalty VALUE drift {val_diff!r} exceeds current ceiling "
        f"{FULL_PENALTY_VALUE_DRIFT_CEILING!r}; regression since the "
        "2026-05-08 baseline of ~3.33e-16."
    )
    assert grad_diff < FULL_PENALTY_GRAD_DRIFT_CEILING, (
        f"Full-penalty GRADIENT drift {grad_diff!r} exceeds current ceiling "
        f"{FULL_PENALTY_GRAD_DRIFT_CEILING!r}; regression since the "
        "2026-05-08 baseline of ~1.29e-13."
    )
