"""Census-level tests for the Boozer derivative input bit-identity ladder.

These tests pin the *current* state of the boundary-input parity ladder, which
``docs/boozer_derivative_bit_identity_impl_plan_2026-05-07.md`` evolves across
Phases 1–3:

* Phase 1 (this slice) — production fast paths still feed the residual; the
  census names ``gamma`` as the first non-byte-identical owner. The
  divergence cascade through ``xphi``, ``xtheta``, the surface coefficient
  Jacobians, and Biot-Savart outputs is recorded as expected.
* Phase 2 — surface CPU-ordered parity twins land; the surface arrays
  become byte-identical under the parity backend mode.
* Phase 3 — Biot-Savart CPU-ordered twins close the rest.

Per the plan §7 acceptance gate, these assertions are *positive* (the test
passes today by reporting the expected first owner) and they are *not*
xfail/skip — the parity sweeps in later phases will tighten them rather than
unmark a hidden failure.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


pytestmark = [
    pytest.mark.parity_census,
]


@pytest.fixture(scope="module")
def synthetic_fixture():
    """Build a paired CPU/JAX BoozerSurface fixture on NCSX coils."""
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from benchmarks.parity.boozer_derivative_input_repro import (
        _build_synthetic_fixture,
    )

    return _build_synthetic_fixture()


@pytest.fixture(scope="module")
def census_records_and_arrays(synthetic_fixture):
    """Capture CPU and JAX boundary records + raw numpy arrays."""
    from benchmarks.parity.boozer_derivative_input_census import (
        capture_cpu_boozer_inputs,
        capture_jax_boozer_inputs,
    )
    from benchmarks.parity.boozer_derivative_input_repro import (
        _materialize_cpu_arrays_dict,
        _materialize_jax_arrays_dict,
    )

    cpu_arrays, cpu_scalars = capture_cpu_boozer_inputs(
        synthetic_fixture["boozer_cpu"],
        sdofs=synthetic_fixture["sdofs"],
        iota=synthetic_fixture["iota"],
        G=synthetic_fixture["G"],
        weight_inv_modB=synthetic_fixture["weight_inv_modB"],
    )
    jax_arrays, jax_scalars = capture_jax_boozer_inputs(
        synthetic_fixture["boozer_jax"],
        sdofs=synthetic_fixture["sdofs"],
        iota=synthetic_fixture["iota"],
        G=synthetic_fixture["G"],
        weight_inv_modB=synthetic_fixture["weight_inv_modB"],
        optimize_G=synthetic_fixture["optimize_G"],
    )
    cpu_arr_dict = _materialize_cpu_arrays_dict(synthetic_fixture)
    jax_arr_dict = _materialize_jax_arrays_dict(synthetic_fixture)
    return {
        "cpu_arrays": cpu_arrays,
        "cpu_scalars": cpu_scalars,
        "jax_arrays": jax_arrays,
        "jax_scalars": jax_scalars,
        "cpu_arr_dict": cpu_arr_dict,
        "jax_arr_dict": jax_arr_dict,
    }


@pytest.fixture(scope="module")
def diffs(census_records_and_arrays):
    from benchmarks.parity.boozer_derivative_input_census import (
        compare_boundary_inputs,
    )

    rd = census_records_and_arrays
    return compare_boundary_inputs(
        cpu_array_records=rd["cpu_arrays"],
        cpu_scalar_records=rd["cpu_scalars"],
        jax_array_records=rd["jax_arrays"],
        jax_scalar_records=rd["jax_scalars"],
        cpu_arrays=rd["cpu_arr_dict"],
        jax_arrays=rd["jax_arr_dict"],
    )


# ---------------------------------------------------------------------------
# Schema and record shape


def test_census_array_record_round_trip():
    from benchmarks.parity.boozer_derivative_input_census import (
        build_array_record,
    )

    arr = np.arange(12, dtype=np.float64).reshape(2, 2, 3)
    rec = build_array_record(
        array_name="gamma",
        producer="cpu",
        stage="boozer_ls_callback_input",
        array=arr,
    )
    payload = rec.to_json_record()
    assert payload["kind"] == "array"
    assert payload["array_name"] == "gamma"
    assert payload["producer"] == "cpu"
    assert payload["dtype"] == "float64"
    assert payload["shape"] == [2, 2, 3]
    assert len(payload["sha256_float64_bytes"]) == 64


def test_census_scalar_record_round_trip():
    from benchmarks.parity.boozer_derivative_input_census import (
        build_scalar_record,
    )

    rec = build_scalar_record(name="iota", producer="cpu", stage="x", value=0.123)
    payload = rec.to_json_record()
    assert payload["kind"] == "scalar"
    assert payload["name"] == "iota"
    assert payload["dtype"] == "float64"
    assert payload["value"] == pytest.approx(0.123)


def test_census_layout_error_on_non_float64():
    from benchmarks.parity.boozer_derivative_input_census import (
        CensusLayoutError,
        build_array_record,
    )

    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    with pytest.raises(CensusLayoutError):
        build_array_record(array_name="bad", producer="cpu", stage="x", array=arr)


def test_cpu_census_producer_preserves_non_float64_evidence():
    from benchmarks.parity.boozer_derivative_input_census import (
        CensusLayoutError,
        _records_from_cpu_inputs,
    )

    inputs = {
        "gamma": np.arange(3, dtype=np.float32),
        "xphi": np.arange(3, dtype=np.float64),
        "xtheta": np.arange(3, dtype=np.float64),
        "dx_dc": np.arange(3, dtype=np.float64),
        "dxphi_dc": np.arange(3, dtype=np.float64),
        "dxtheta_dc": np.arange(3, dtype=np.float64),
        "B": np.arange(3, dtype=np.float64),
        "dB_dx": np.arange(3, dtype=np.float64),
    }

    with pytest.raises(CensusLayoutError, match="cpu::gamma dtype=float32"):
        _records_from_cpu_inputs(
            inputs,
            G=1.0,
            iota=0.25,
            weight_inv_modB=False,
            stage="x",
        )


def test_compare_array_reports_dtype_mismatch_without_casting():
    from benchmarks.parity.boozer_derivative_input_census import compare_array

    diff = compare_array(
        array_name="gamma",
        stage="x",
        cpu=np.asarray([1.0], dtype=np.float64),
        jax_=np.asarray([1.0], dtype=np.float32),
    )

    assert diff.shape_match
    assert not diff.dtype_match
    assert not diff.byte_identical
    assert diff.n_bit_different_entries == -1
    assert np.isnan(diff.max_abs_diff)


# ---------------------------------------------------------------------------
# Production-mode ladder shape (Phase 1 expectation)


def test_census_records_cover_canonical_ladder(census_records_and_arrays):
    from benchmarks.parity.boozer_derivative_input_census import (
        CENSUS_BOUNDARY_ARRAY_ORDER,
        CENSUS_BOUNDARY_SCALAR_ORDER,
    )

    rd = census_records_and_arrays
    cpu_array_names = {r.array_name for r in rd["cpu_arrays"]}
    jax_array_names = {r.array_name for r in rd["jax_arrays"]}
    assert set(CENSUS_BOUNDARY_ARRAY_ORDER).issubset(cpu_array_names)
    assert set(CENSUS_BOUNDARY_ARRAY_ORDER).issubset(jax_array_names)

    cpu_scalar_names = {r.name for r in rd["cpu_scalars"]}
    jax_scalar_names = {r.name for r in rd["jax_scalars"]}
    assert set(CENSUS_BOUNDARY_SCALAR_ORDER).issubset(cpu_scalar_names)
    assert set(CENSUS_BOUNDARY_SCALAR_ORDER).issubset(jax_scalar_names)


def test_first_divergence_is_gamma_in_production_mode(diffs):
    """Until Phase 2 wires CPU-ordered surface twins, ``gamma`` is the first
    array to drift. The plan's Lane 6 ladder predicts this; the test pins it
    so any reordering of the divergence ladder is caught in review.
    """
    from benchmarks.parity.boozer_derivative_input_census import (
        CensusArrayDiff,
        first_divergence,
    )

    fd = first_divergence(diffs)
    assert isinstance(fd, CensusArrayDiff), (
        "Phase 1 expects an array-typed first divergence; scalars (G, iota, "
        "weight_inv_modB) are constants in this contract."
    )
    assert fd.array_name == "gamma", (
        f"Expected the first divergent owner to be 'gamma'; got "
        f"{fd.array_name!r}. If a Phase 2/3 substitution closed gamma, this "
        "test should be tightened (assert byte_identity for gamma + xphi + "
        "xtheta) instead of left as-is."
    )
    assert not fd.byte_identical


def test_scalar_inputs_byte_identical_in_production_mode(diffs):
    from benchmarks.parity.boozer_derivative_input_census import (
        CensusScalarDiff,
    )

    scalar_diffs = [d for d in diffs if isinstance(d, CensusScalarDiff)]
    assert scalar_diffs, "expected scalar diffs for G, iota, weight_inv_modB"
    for d in scalar_diffs:
        assert d.byte_identical, (
            f"scalar boundary input {d.name} unexpectedly diverged "
            f"({d.cpu_value} vs {d.jax_value}); G and iota are constants "
            "and must stay bit-identical regardless of parity mode."
        )


def test_array_diffs_record_max_abs_diff_within_ulp_floor(diffs):
    """Sanity check on diff magnitudes — Phase 1 expects sub-ULP for gamma."""
    from benchmarks.parity.boozer_derivative_input_census import (
        CensusArrayDiff,
    )

    by_name = {d.array_name: d for d in diffs if isinstance(d, CensusArrayDiff)}
    gamma = by_name["gamma"]
    # ``gamma`` is order-1 in magnitude on the NCSX fixture; sub-ULP drift
    # under double precision is below ~1e-15.
    assert 0.0 < gamma.max_abs_diff < 1e-13, (
        f"unexpected gamma drift magnitude: {gamma.max_abs_diff!r}"
    )


def test_ndjson_emission_stable(tmp_path, census_records_and_arrays, diffs):
    from benchmarks.parity.boozer_derivative_input_census import write_ndjson

    rd = census_records_and_arrays
    out = write_ndjson(
        tmp_path / "census.ndjson",
        list(rd["cpu_arrays"])
        + list(rd["jax_arrays"])
        + list(rd["cpu_scalars"])
        + list(rd["jax_scalars"])
        + list(diffs),
    )
    text = out.read_text().splitlines()
    assert len(text) == (
        len(rd["cpu_arrays"])
        + len(rd["jax_arrays"])
        + len(rd["cpu_scalars"])
        + len(rd["jax_scalars"])
        + len(diffs)
    )
    # Every line must round-trip as JSON with a ``kind`` discriminator.
    import json

    for line in text:
        rec = json.loads(line)
        assert "kind" in rec
