"""The flat675 -> nested bridge must hand over the SAME problem, not a neighbour.

Everything the endpoint comparison concludes rests on one equality: the native
and JAX objects :func:`nested_view_from_flat675` builds must reproduce the
flat-675 material's own field, surface and inner state at the vector they were
built from.  A bridge that is off by a coil source, a quadrature, or a DOF
ordering would still produce a plausible Boozer residual — of a different
problem — so each of those three is pinned here by a number rather than by a
smoke check.

Both configurations are covered, because they exercise different reconstruction
paths: repository geometry runs the symmetric-replica path (``RotatedCurve``
matrices and ``ScaledCurrent`` factors recovered from the extraction spec, with
one base curve shared by every replica), and the frozen bundle runs the
archived twenty-fixed-TF-coil layout at the certified quadrature.

The bundle is host-local.  When its directory is absent this file FAILS rather
than skipping: a silent skip is how a bridge that never ran against the
certified configuration comes to look certified.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from simsopt._core.json import GSONDecoder
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.geo import (
    SurfaceRZFourier,
    Volume,
    create_equally_spaced_curves,
)
from simsopt_jax.core.field import grouped_biot_savart_B_from_spec
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.geo import CurveCWSFourier
from simsopt_jax_adapters.geo.flat675 import (
    Flat675Problem,
    build_flat675_problem,
)
from simsopt_jax_adapters.geo.flat675.boozer_material import (
    build_flat675_boozer_system,
    flat675_candidate_geometry,
)
from simsopt_jax_adapters.geo.flat675.bundle import load_flat675_bundle
from simsopt_jax_adapters.geo.flat675.nested_bridge import (
    NESTED_BRIDGE_NEWTON_OPTIONS,
    Flat675NestedBridgeError,
    nested_view_from_flat675,
)
from simsopt_jax_adapters.geo.nested_ls_contract import (
    NESTED_LS_BANANA_NEWTON_MAXITER,
    NESTED_LS_BANANA_NEWTON_TOL,
    NESTED_LS_CONSTRAINT_WEIGHT,
    NESTED_LS_WEIGHT_INV_MODB,
)
from simsopt_jax_adapters.geo.flat675.y_solve import solve_flat675_y_qr

# The frozen campaign input. The environment variable lets a host point the
# gate at a relocated copy; it never turns the gate off.
_BUNDLE_ENV_VAR = "SIMSOPT_FLAT675_BUNDLE_ROOT"
_DEFAULT_BUNDLE_ROOT = (
    Path.home() / "simsopt_mixed_artifacts" / "genuine675-r3-input-1c23f6c5-20260721-r1"
)
_NATIVE_BIOT_SAVART_FILENAME = "native_biot_savart.json"

# Repository geometry, at the smallest scale that still exercises every
# reconstruction path. Nothing here is a physics claim.
_BOUNDARY_INPUT = (
    Path(__file__).resolve().parents[3]
    / "tests"
    / "test_files"
    / "input.LandremanPaul2021_QA_lowres"
)
_GRID = 6
_CURVE_QUADPOINTS = 24
_CURVE_ORDER = 2
_WINDING_SURFACE_FACTOR = 2.2
_TF_COIL_RADIUS_FACTOR = 2.6
_TF_BASE_COIL_COUNT = 3
_TF_COIL_CURRENT_A = 1.0e5
_WINDING_COIL_CURRENT_A = 1.5e5
_WINDING_COIL_DOFS = (
    0.119251,
    0.012469,
    -0.017700,
    -0.068677,
    -0.014250,
    0.430936,
    0.082708,
    -0.015905,
    -0.113852,
    -0.025142,
)

# The bridge reconstructs native coils through a different arithmetic route
# than the JAX kernels (simsopt's C++ Biot-Savart against XLA's), so the two
# fields agree to round-off on the field scale rather than bitwise. This bound
# is three orders tighter than the contract's 1e-12 and two orders above the
# observed value, so it fails on a real divergence and not on a compiler.
_FIELD_RTOL = 1.0e-13


class CertifiedBundleMissing(RuntimeError):
    """The certified frozen bundle is absent, so the bundle cases FAIL.

    Not an ``assert``: ``assert`` statements are stripped under ``python -O``,
    and a gate whose only environment precondition disappears under an
    interpreter flag is a gate that can be switched off by the way it is run.
    Not a skip either, for the reason in this module's docstring.
    """


def _bundle_root() -> Path:
    root = Path(os.environ.get(_BUNDLE_ENV_VAR, str(_DEFAULT_BUNDLE_ROOT)))
    if not root.is_dir():
        raise CertifiedBundleMissing(
            f"the certified frozen bundle was not found at {root}. This gate "
            "covers the certified configuration and does not skip: point "
            f"{_BUNDLE_ENV_VAR} at the bundle, or run this file on a host that "
            "has it."
        )
    return root


def _repository_problem() -> Flat675Problem:
    """The JAX example's repository-geometry problem, at bounded scale."""
    boundary = SurfaceRZFourier.from_vmec_input(
        str(_BOUNDARY_INPUT), range="half period", nphi=_GRID, ntheta=_GRID
    )
    points = np.asarray(boundary.gamma(), dtype=np.float64).reshape((-1, 3))
    radius = np.hypot(points[:, 0], points[:, 1])
    major_radius = 0.5 * (float(radius.max()) + float(radius.min()))
    minor_radius = float(np.max(np.hypot(radius - major_radius, points[:, 2])))

    winding_surface = SurfaceRZFourier(
        nfp=boundary.nfp,
        stellsym=True,
        mpol=1,
        ntor=0,
        quadpoints_phi=np.linspace(0.0, 1.0, 16, endpoint=False),
        quadpoints_theta=np.linspace(0.0, 1.0, 16, endpoint=False),
    )
    winding_surface.set_rc(0, 0, major_radius)
    winding_surface.set_rc(1, 0, minor_radius * _WINDING_SURFACE_FACTOR)
    winding_surface.set_zs(1, 0, minor_radius * _WINDING_SURFACE_FACTOR)

    base_curve = CurveCWSFourier(
        quadpoints=_CURVE_QUADPOINTS, order=_CURVE_ORDER, surf=winding_surface
    )
    base_curve.x = np.asarray(_WINDING_COIL_DOFS, dtype=np.float64)
    winding_coils = coils_via_symmetries(
        [base_curve], [Current(_WINDING_COIL_CURRENT_A)], boundary.nfp, True
    )

    tf_curves = create_equally_spaced_curves(
        _TF_BASE_COIL_COUNT,
        boundary.nfp,
        stellsym=True,
        R0=major_radius,
        R1=minor_radius * _TF_COIL_RADIUS_FACTOR,
        order=_CURVE_ORDER,
        numquadpoints=32,
        use_jax_curve=False,
    )
    tf_currents = []
    for curve in tf_curves:
        curve.fix_all()
        current = Current(_TF_COIL_CURRENT_A)
        current.fix_all()
        tf_currents.append(current)
    tf_coils = coils_via_symmetries(tf_curves, tf_currents, boundary.nfp, True)

    field = BiotSavartJAX(list(tf_coils) + list(winding_coils))
    return build_flat675_problem(
        boundary=boundary, field=field, nphi=_GRID, ntheta=_GRID
    )


@pytest.fixture(scope="module")
def repository_problem() -> Flat675Problem:
    return _repository_problem()


@pytest.fixture(scope="module")
def bundle_problem() -> Flat675Problem:
    return load_flat675_bundle(_bundle_root())


@pytest.fixture(params=["repository-geometry", "bundle"])
def problem(request) -> Flat675Problem:
    # Resolve lazily so the repository-geometry cases never touch the bundle
    # fixture: a host without the frozen bundle still runs half this file.
    if request.param == "bundle":
        return request.getfixturevalue("bundle_problem")
    return request.getfixturevalue("repository_problem")


def _perturbed_vector(problem: Flat675Problem) -> np.ndarray:
    """A vector that is NOT the start, so nothing can pass by reading the seed.

    The perturbation is deterministic and small enough to stay in the same
    geometric regime, but it moves every block, which is what makes the coil
    block a real input to the reconstruction rather than a constant.
    """
    vector = np.asarray(problem.start_candidate.outer_vector(), dtype=np.float64)
    offsets = 1.0e-3 * np.cos(np.arange(vector.size, dtype=np.float64))
    return vector + offsets


def _material_field_and_gamma(
    problem: Flat675Problem,
    vector: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """The flat-675 material's own field on its own surface quadrature."""
    layout = problem.material.layout
    geometry = flat675_candidate_geometry(
        problem.material.boozer,
        jnp.asarray(vector[layout.coil_slice], dtype=jnp.float64),
        jnp.asarray(vector[layout.surface_slice], dtype=jnp.float64),
    )
    gamma = np.asarray(geometry.surface_gamma, dtype=np.float64).reshape((-1, 3))
    field = np.asarray(
        grouped_biot_savart_B_from_spec(jnp.asarray(gamma), geometry.coil_set),
        dtype=np.float64,
    )
    return field, gamma


def _relative_field_error(reference: np.ndarray, other: np.ndarray) -> float:
    return float(
        np.max(np.linalg.norm(other - reference, axis=1))
        / np.max(np.linalg.norm(reference, axis=1))
    )


def test_native_field_reproduces_the_flat675_material(problem) -> None:
    """Guarantee 1: same coils, to round-off on the field scale."""
    vector = _perturbed_vector(problem)
    view = nested_view_from_flat675(problem, vector)
    material_field, gamma = _material_field_and_gamma(problem, vector)

    view.biotsavart_native.set_points(gamma)
    error = _relative_field_error(
        material_field, np.asarray(view.biotsavart_native.B(), dtype=np.float64)
    )
    assert error < _FIELD_RTOL, f"native field differs from the material by {error!r}"

    view.jax_inputs.biotsavart_jax.set_points(gamma)
    jax_error = _relative_field_error(
        material_field,
        np.asarray(view.jax_inputs.biotsavart_jax.B(), dtype=np.float64),
    )
    assert jax_error < _FIELD_RTOL, f"jax field differs by {jax_error!r}"


def test_native_surface_is_the_vectors_surface_block(problem) -> None:
    """Guarantee 2: bitwise DOFs, and the material's own quadrature."""
    vector = _perturbed_vector(problem)
    view = nested_view_from_flat675(problem, vector)
    template = problem.material.boozer.surface_template

    assert np.array_equal(
        np.asarray(view.surface_native.get_dofs(), dtype=np.float64),
        vector[problem.material.layout.surface_slice],
    )
    assert np.array_equal(
        np.asarray(view.surface_native.quadpoints_phi, dtype=np.float64),
        np.asarray(template.quadpoints_phi, dtype=np.float64),
    )
    assert np.array_equal(
        np.asarray(view.surface_native.quadpoints_theta, dtype=np.float64),
        np.asarray(template.quadpoints_theta, dtype=np.float64),
    )
    assert int(view.surface_native.mpol) == int(template.mpol)
    assert int(view.surface_native.ntor) == int(template.ntor)

    # The native surface and the traced geometry must be the same points, or
    # the nested residual is evaluated somewhere the flat route never looked.
    _material_field, gamma = _material_field_and_gamma(problem, vector)
    points_error = float(
        np.max(
            np.abs(
                np.asarray(view.surface_native.gamma(), dtype=np.float64).reshape(
                    (-1, 3)
                )
                - gamma
            )
        )
    )
    assert points_error < 1.0e-12, f"surface points differ by {points_error!r} m"


def test_iota_and_G_are_the_flat675_y_solve(problem) -> None:
    """Guarantee 3: the inner state is the flat route's, bitwise."""
    vector = _perturbed_vector(problem)
    view = nested_view_from_flat675(problem, vector)
    layout = problem.material.layout
    geometry = flat675_candidate_geometry(
        problem.material.boozer,
        jnp.asarray(vector[layout.coil_slice], dtype=jnp.float64),
        jnp.asarray(vector[layout.surface_slice], dtype=jnp.float64),
    )
    system = build_flat675_boozer_system(geometry, problem.boozer_policy)
    expected = np.asarray(
        solve_flat675_y_qr(system.design_matrix, system.right_hand_side).solution,
        dtype=np.float64,
    )
    assert view.iota == float(expected[0])
    assert view.G == float(expected[1])


def test_wrong_length_vector_is_refused(problem) -> None:
    """Guarantee 4: a typed refusal, not a broadcast."""
    outer = problem.material.layout.outer_dof_count
    for length in (outer - 1, outer + 1):
        with pytest.raises(Flat675NestedBridgeError):
            nested_view_from_flat675(problem, np.zeros(length, dtype=np.float64))


def test_blocks_are_the_vectors_own_and_are_copies(problem) -> None:
    vector = _perturbed_vector(problem)
    layout = problem.material.layout
    view = nested_view_from_flat675(problem, vector)

    assert np.array_equal(view.coil_dofs, vector[layout.coil_slice])
    assert np.array_equal(view.vessel_dofs, vector[layout.vessel_slice])
    assert np.array_equal(view.surface_dofs, vector[layout.surface_slice])
    assert np.array_equal(view.vector, vector)

    vector[layout.coil_slice][0] += 1.0
    vector[layout.vessel_slice][0] += 1.0
    vector[layout.surface_slice][0] += 1.0
    assert not np.array_equal(view.vector, vector)
    assert view.coil_dofs[0] != vector[layout.coil_slice][0]
    assert view.vessel_dofs[0] != vector[layout.vessel_slice][0]
    assert view.surface_dofs[0] != vector[layout.surface_slice][0]


def test_jax_inputs_hand_out_a_fresh_solver_object(problem) -> None:
    """B's two lanes must not inherit each other's mutated surface."""
    vector = _perturbed_vector(problem)
    view = nested_view_from_flat675(problem, vector)

    first = view.jax_inputs.new_boozer_surface_jax()
    second = view.jax_inputs.new_boozer_surface_jax()
    assert first is not second
    assert first.surface is not second.surface
    assert first.biotsavart is view.jax_inputs.biotsavart_jax
    assert isinstance(first.label, Volume)
    assert float(first.targetlabel) == float(
        problem.objective_policy.boozer_target_label
    )

    first.surface.set_dofs(np.zeros_like(view.surface_dofs))
    assert np.array_equal(
        np.asarray(view.jax_inputs.new_boozer_surface_jax().surface.get_dofs()),
        view.surface_dofs,
    )
    assert np.array_equal(
        np.asarray(view.surface_native.get_dofs(), dtype=np.float64),
        view.surface_dofs,
    )


def test_the_bridge_hands_over_the_timing_bar_newton_policy(problem) -> None:
    """Guarantee 4: the inner-solver policy is the banana (TIMING) one.

    The endpoint comparison judges both of its lanes at the nested route's
    banana ``run_code`` policy, and the view it judges is built here, so these
    two numbers ARE that bar.  They were the reconstruct/physics pair once;
    pinning them against the contract constants is what stops the bridge and
    the harness from drifting back apart silently.
    """
    vector = _perturbed_vector(problem)
    view = nested_view_from_flat675(problem, vector)
    options = view.jax_inputs.newton_options
    assert options is NESTED_BRIDGE_NEWTON_OPTIONS
    assert options["newton_tol"] == NESTED_LS_BANANA_NEWTON_TOL
    assert options["newton_maxiter"] == NESTED_LS_BANANA_NEWTON_MAXITER
    assert options["weight_inv_modB"] == NESTED_LS_WEIGHT_INV_MODB
    assert float(view.jax_inputs.constraint_weight) == float(
        NESTED_LS_CONSTRAINT_WEIGHT
    )


def test_native_label_is_over_the_views_own_surface(problem) -> None:
    vector = _perturbed_vector(problem)
    view = nested_view_from_flat675(problem, vector)
    assert isinstance(view.label_native, Volume)
    assert view.label_native.surface is view.surface_native


def test_bundle_coil_source_is_the_runtime_spec(bundle_problem) -> None:
    """The archived Biot-Savart file is never the bridge's coil source.

    The orchestrator's contract flags the ``single_stage_seed_iota15`` class of
    defect, where a fixture's runtime spec and its Biot-Savart JSON carry
    different coil states. This asserts the property that makes the bridge
    immune to it — the reconstruction is a function of the material's
    extraction spec and the vector's coil block alone — and records what the
    archived file actually says on THIS bundle, which is a separate fact from
    whether the bridge reads it.
    """
    root = _bundle_root()
    vector = _perturbed_vector(bundle_problem)
    view = nested_view_from_flat675(bundle_problem, vector)
    material_field, gamma = _material_field_and_gamma(bundle_problem, vector)

    archived = json.loads(
        (root / _NATIVE_BIOT_SAVART_FILENAME).read_text(), cls=GSONDecoder
    )
    assert isinstance(archived, BiotSavart)

    # What the archive is frozen at. On THIS bundle it is the start candidate's
    # coil block bitwise -- the two sources agree, unlike the iota15 fixture --
    # and that is worth pinning: a future bundle where they diverge would
    # silently split the native twin from the JAX lane, exactly as the flagged
    # defect did, without touching the bridge.
    assert np.array_equal(
        np.asarray(archived.x, dtype=np.float64),
        np.asarray(bundle_problem.start_candidate.coil_coordinates, dtype=np.float64),
    ), (
        "the bundle's native_biot_savart.json is no longer frozen at the coil "
        "block its runtime spec carries; consumers that read the archive and "
        "consumers that read the spec now solve different problems."
    )

    # Driven to the vector's own coil block, which is how a consumer that DID
    # read the archive would use it.
    archived.x = np.array(view.coil_dofs, dtype=np.float64)
    archived.set_points(gamma)
    driven = _relative_field_error(
        material_field, np.asarray(archived.B(), dtype=np.float64)
    )

    # The bridge's own field is the reference either way: it is built from the
    # extraction spec, so driving or not driving the archive cannot move it.
    view.biotsavart_native.set_points(gamma)
    bridge_error = _relative_field_error(
        material_field, np.asarray(view.biotsavart_native.B(), dtype=np.float64)
    )
    assert bridge_error < _FIELD_RTOL
    assert driven < _FIELD_RTOL, (
        "the archived native_biot_savart.json no longer agrees with the "
        f"runtime spec at the vector's coil block ({driven!r}); the bridge is "
        "unaffected, but every consumer that reads the archive now solves a "
        "different problem."
    )
