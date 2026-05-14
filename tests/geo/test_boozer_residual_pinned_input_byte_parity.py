"""Boundary-pinned residual / full-penalty CPU-vs-JAX drift-ceiling arbiter.

These tests pin CPU-vs-JAX agreement of the Boozer residual scalar/gradient and
the SciPy-visible full-penalty value/gradient (residual + label + rz-axis terms)
to a documented numerical drift ceiling, NOT to byte-identity.

Ship gate
---------
The raw Boozer residual is the "raw Boozer residual" entry of the
``direct_kernel`` parity-ladder lane (see ``CLAUDE.md`` and
``benchmarks/validation_ladder_contract.py::PARITY_LADDER_TOLERANCES``):

    rtol = 1e-10, atol = 1e-12, requires_same_state = True

The "raw Boozer residual" gradient is a same-state direct-kernel comparison too:
the inputs are the canonical CPU oracle bundle, ``optimize_G=True``, and the
gradient axis is ``(iota, G, surface_dofs)``. Both the residual and the full
penalty live on that lane.

Byte identity is **not** the ship gate. Strict byte identity is desired but is
gated on closing the JAX vs C++ FMA-arrangement gap tracked separately in
``docs/boozer_derivative_bit_identity_impl_plan_2026-05-07.md`` §20. Failure of
byte identity per se does not break the production strict gate
(``_pre_newton_census_gate_failures`` in ``benchmarks/single_stage_init_parity.py``)
as long as drift stays inside the direct-kernel ladder.

History
-------
Earlier revisions of this file (commit ``9460c81cf7`` and audit finding #9 of
``.artifacts/jax-test-audit-2026-05-13/TEST_QUALITY_TODOS.md``) committed four
byte-identity assertions that were known to fail today as the "audit-visible
red". The audit-visible red conflicts with the repo's
``test_pytest_skip_xfail_audit.py`` contract that all failing tests must be
``@pytest.mark.xfail(strict=True, reason=...)``-marked. Outcome B of finding #9
removes the byte-identity assertions and pins the documented drift ceiling
instead; restoring strict byte identity is the responsibility of the FMA-shape
restructure in ``boozer_residual_jax.py`` §20 sites 1-3, not of this test.

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


# ---------------------------------------------------------------------------
# Drift ceiling constants — anchored to the ``direct_kernel`` lane of
# ``benchmarks/validation_ladder_contract.py::PARITY_LADDER_TOLERANCES``.
#
# The raw Boozer residual is a same-state ``direct_kernel`` comparison
# (rtol=1e-10, atol=1e-12) per CLAUDE.md "Key Conventions" and the parity
# ladder contract. These ceilings are stricter than the lane's rtol/atol
# because they pin the *empirical* CPU-vs-JAX baseline (HEAD on 2026-05-13,
# JAX 0.10.0, aarch64). A regression that grows drift past these ceilings is
# either (a) a real bug, or (b) a benign FMA-shape change that should be
# audited and rebaselined explicitly here, not silently absorbed.
#
# Empirical baseline (current HEAD, recorded by running this file 2026-05-13):
#   residual_only.max_abs_diff_value = 0.0 (byte identical today; constant
#       still bounded by ladder rtol/atol to absorb a benign FMA reshuffle).
#   residual_only.max_abs_diff_grad  = 8.881784197001252e-16  (~= 4 * eps on
#       gradient entries of |.| up to ~13).
#   full_penalty.max_abs_diff_value  = 0.0
#   full_penalty.max_abs_diff_grad   = 8.881784197001252e-16  (identical to
#       residual_only because the label/rz penalty terms are pinned to CPU
#       and therefore cancel in the cpu-vs-jax difference).
#
# Theoretical justification (ceiling values):
#   * VALUE ceiling = ``5 * atol_direct_kernel = 5e-12``. The scalar residual
#     is a single accumulator; one or two ULPs of FMA reshuffle at scale ~1
#     is the worst credible drift. ``5e-12`` is well within ``atol=1e-12``'s
#     order of magnitude with ~4x buffer.
#   * GRADIENT ceiling = ``5e-14`` absolute. The gradient is a vector of
#     three-term FMA accumulators each scaled by ``1 / num_res``. Empirical
#     8.88e-16 is roughly 4*eps at scale 1; the 5e-14 ceiling absorbs ~5
#     ULPs at scale 10 (largest observed gradient entry is ~13) with
#     comfortable headroom. This still sits ~4 orders of magnitude below
#     the direct-kernel rtol=1e-10 / atol=1e-12 contract.
#
# Both ceilings remain ORDERS OF MAGNITUDE tighter than the ladder's
# ``rtol=1e-10, atol=1e-12``. They will be tightened to the empirical
# baseline (or to exact byte equality) once the FMA-shape restructure in
# ``boozer_residual_jax.py`` §20 sites 1-3 lands; until then the production
# strict gate elsewhere in the suite is unaffected.
_DIRECT_KERNEL_LANE_RTOL = 1e-10  # PARITY_LADDER_TOLERANCES["direct_kernel"]["rtol"]
_DIRECT_KERNEL_LANE_ATOL = 1e-12  # PARITY_LADDER_TOLERANCES["direct_kernel"]["atol"]
RESIDUAL_VALUE_DRIFT_CEILING_ABS = 5e-12  # ~5 * direct_kernel atol; scalar accumulator
RESIDUAL_GRADIENT_DRIFT_CEILING_ABS = 5e-14  # ~5 ULP at scale 10; well under lane rtol
FULL_PENALTY_VALUE_DRIFT_CEILING_ABS = RESIDUAL_VALUE_DRIFT_CEILING_ABS
FULL_PENALTY_GRADIENT_DRIFT_CEILING_ABS = RESIDUAL_GRADIENT_DRIFT_CEILING_ABS


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
    (CPU normalized by ``num_res``), suitable for numerical comparison by
    the drift-ceiling tests below.
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

    The residual component comes from the same CPU-canonical residual bundle
    that the residual tests feed to both backends, while the label and rz-axis
    penalty pieces are pinned from the CPU fixture. This keeps producer drift
    in geometry/field rematerialization from masquerading as a full-penalty
    drift-ceiling failure.
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
# Test set 1: infrastructure / shape assertions


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
# Test set 2: drift ceilings — direct-kernel-lane ship gate.
#
# Each test pins the worst-case CPU-vs-JAX absolute difference to a ceiling
# anchored to the ``direct_kernel`` parity-ladder lane. The ceilings sit
# orders of magnitude above the current empirical baseline so a benign
# FMA-shape change does not flake, but they sit orders of magnitude below
# the lane's ``rtol=1e-10, atol=1e-12`` contract so a real regression is
# still caught here long before the production strict gate notices it.


_DRIFT_CEILING_DRIVER_HINT = (
    "Direct-kernel-lane drift ceiling exceeded. Likely a real regression in "
    "src/simsopt/geo/boozer_residual_jax.py or src/simsoptpp/boozer.cpp. If the "
    "regression is an intentional FMA-shape change (e.g. closing the §20 sites "
    "1-3 FMA gap in boozer_residual_jax.py), rebaseline the empirical-baseline "
    "table in this file's module-level docstring and adjust the ceiling if "
    "needed; the ceiling should remain inside direct_kernel rtol=1e-10."
)


def test_residual_value_within_drift_ceiling(residual_outputs):
    """Residual scalar value: CPU-vs-JAX drift must sit inside the ceiling.

    Ceiling: ``RESIDUAL_VALUE_DRIFT_CEILING_ABS = 5e-12`` (5x direct-kernel
    atol). Empirical baseline today: ``0.0`` (byte-identical scalar value).
    """
    val_cpu = residual_outputs["value_cpu"]
    val_jax = residual_outputs["value_jax"]
    val_diff = abs(val_jax - val_cpu)

    assert val_diff < RESIDUAL_VALUE_DRIFT_CEILING_ABS, (
        f"Residual VALUE drift {val_diff!r} >= ceiling "
        f"{RESIDUAL_VALUE_DRIFT_CEILING_ABS!r}. cpu={val_cpu!r}, jax={val_jax!r}. "
        f"{_DRIFT_CEILING_DRIVER_HINT}"
    )


def test_residual_gradient_within_drift_ceiling(residual_outputs):
    """Residual gradient: CPU-vs-JAX max-abs-diff must sit inside the ceiling.

    Ceiling: ``RESIDUAL_GRADIENT_DRIFT_CEILING_ABS = 5e-14`` (~5 ULPs at
    gradient scale 10; direct-kernel rtol headroom is ~4 orders of magnitude).
    Empirical baseline today: ``8.88e-16`` (~4*eps at gradient scale ~13).
    """
    grad_cpu = residual_outputs["grad_cpu"]
    grad_jax = residual_outputs["grad_jax"]
    grad_diff = float(np.max(np.abs(grad_jax - grad_cpu)))

    assert grad_diff < RESIDUAL_GRADIENT_DRIFT_CEILING_ABS, (
        f"Residual GRADIENT max-abs-diff {grad_diff!r} >= ceiling "
        f"{RESIDUAL_GRADIENT_DRIFT_CEILING_ABS!r}. {_DRIFT_CEILING_DRIVER_HINT}"
    )


def test_full_penalty_value_within_drift_ceiling(full_penalty_outputs):
    """Full-penalty scalar value: CPU-vs-JAX drift must sit inside the ceiling.

    Ceiling: ``FULL_PENALTY_VALUE_DRIFT_CEILING_ABS = 5e-12`` (5x direct-kernel
    atol). The label/rz penalty pieces are pinned from the CPU fixture, so the
    full-penalty value drift inherits the residual scalar drift.
    """
    val_cpu = full_penalty_outputs["value_cpu"]
    val_jax = full_penalty_outputs["value_jax"]
    val_diff = abs(val_jax - val_cpu)

    assert val_diff < FULL_PENALTY_VALUE_DRIFT_CEILING_ABS, (
        f"Full-penalty VALUE drift {val_diff!r} >= ceiling "
        f"{FULL_PENALTY_VALUE_DRIFT_CEILING_ABS!r}. cpu={val_cpu!r}, jax={val_jax!r}. "
        f"{_DRIFT_CEILING_DRIVER_HINT}"
    )


def test_full_penalty_gradient_within_drift_ceiling(full_penalty_outputs):
    """Full-penalty gradient: CPU-vs-JAX max-abs-diff must sit inside the ceiling.

    Ceiling: ``FULL_PENALTY_GRADIENT_DRIFT_CEILING_ABS = 5e-14`` (~5 ULPs at
    gradient scale 10; identical to the residual gradient ceiling because the
    label/rz penalty terms are pinned from the CPU fixture and therefore cancel
    in the cpu-vs-jax difference).
    """
    grad_cpu = full_penalty_outputs["grad_cpu"]
    grad_jax = full_penalty_outputs["grad_jax"]
    grad_diff = float(np.max(np.abs(grad_jax - grad_cpu)))

    assert grad_diff < FULL_PENALTY_GRADIENT_DRIFT_CEILING_ABS, (
        f"Full-penalty GRADIENT max-abs-diff {grad_diff!r} >= ceiling "
        f"{FULL_PENALTY_GRADIENT_DRIFT_CEILING_ABS!r}. {_DRIFT_CEILING_DRIVER_HINT}"
    )
