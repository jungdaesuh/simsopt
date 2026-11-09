"""JaxBiotSavart and its kernels compute what native BiotSavart computes.

Fields, vector potentials, their point derivatives, the per-coil unit-current
fields and the coil-DOF pullbacks are compared with native simsoptpp
``BiotSavart``, with analytic fields and with shrinking-step Taylor tests, for
Fourier, symmetric, framed, filament and perturbed curves and composed
currents. The tiled kernels (coil chunks, quadrature blocks, point chunks) are
compared with the untiled ones, and their compiled programs are checked for
the forms that keep them fast. The coil-DOF extraction maps are compared
bitwise with the native DOF setters, and a short stage-two optimization runs
the native objectives on the JAX field.
"""

from __future__ import annotations

from unittest_jax_support import JAX_IMPORT_ERROR, JaxTestCase, optional_jax_imports

with optional_jax_imports():
    import simsopt_jax  # noqa: F401
    import jax
    import jax.numpy as jnp
    from jax.extend.core import jaxprs_in_params
    from simsopt_jax.backend import set_backend
    from simsopt_jax.backend.jit import jit
    from simsopt_jax.backend.runtime import disallow_host_transfers
    from simsopt_jax.core import biotsavart as core_biotsavart
    from simsopt_jax.core import (
        coil_specs_from_dof_extraction_spec,
        curve_gamma_and_gammadash_from_spec,
    )
    from simsopt_jax.core.field import (
        group_biot_savart_B_vjp,
        grouped_biot_savart_B_from_spec,
        grouped_coil_set_spec_from_inputs,
    )
    from simsopt_jax.core.specs import CoilGroupSpec, GroupedCoilSetSpec
    from simsopt_jax_adapters.field.biotsavart_backend import (
        BiotSavartFieldPullback,
        JaxBiotSavart,
        _owner_segments_from_free_positions,
        _per_coil_unit_field,
        _per_coil_unit_field_with_batch_size,
    )
    from simsopt_jax_adapters.geo.curve_specs import curve_spec_from_adapter_curve

import re
from dataclasses import dataclass
from typing import cast

import numpy as np
from scipy.optimize import minimize

from simsopt._core.derivative import Derivative
from simsopt._core.optimizable import Optimizable
from simsopt.configs import get_data
from simsopt.field import BiotSavart, Coil, Current, coils_via_symmetries
from simsopt.geo import (
    CurveLength,
    CurveXYZFourier,
    SurfaceRZFourier,
    create_equally_spaced_curves,
)
from simsopt.geo.curveperturbed import (
    CurvePerturbed,
    GaussianSampler,
    PerturbationSample,
)
from simsopt.geo.curvexyzfouriersymmetries import CurveXYZFourierSymmetries
from simsopt.geo.finitebuild import CurveFilament
from simsopt.geo.framedcurve import (
    FrameRotation,
    FramedCurveCentroid,
    FramedCurveFrenet,
)
from simsopt.objectives import SquaredFlux


# Native and analytic references on the NCSX coils and the native Taylor fixture.

if JAX_IMPORT_ERROR is None:
    biot_savart_B = core_biotsavart.biot_savart_B
    biot_savart_dB_by_dX = core_biotsavart.biot_savart_dB_by_dX
    biot_savart_A = core_biotsavart.biot_savart_A
    biot_savart_dA_by_dX = core_biotsavart.biot_savart_dA_by_dX


_DIRECT_KERNEL_TOLS = {
    "rtol": 1e-10,
    "atol": 1e-12,
}


_DERIVATIVE_HEAVY_TOLS = {
    "first_derivative_rtol": 1e-08,
    "first_derivative_atol": 1e-10,
    "second_derivative_rtol": 1e-10,
    "second_derivative_atol": 1e-12,
}


def _ncsx_biotsavart_parity_fixture():

    curves, currents_objs, _, nfp, _ = get_data("ncsx")
    coils = coils_via_symmetries(curves, currents_objs, nfp, stellsym=True)
    bs = BiotSavart(coils)

    npoints = 50
    rng = np.random.default_rng(42)
    points_np = rng.standard_normal((npoints, 3)) * 0.3
    points_np[:, 0] += 1.0  # shift near torus

    bs.set_points(points_np)
    gammas_np = np.array([coil.curve.gamma() for coil in coils])
    gds_np = np.array([coil.curve.gammadash() for coil in coils])
    currents_np = np.array([coil.current.get_value() for coil in coils])
    return bs, points_np, gammas_np, gds_np, currents_np


def _native_taylor_fixture():
    """Coil and points of the native ``tests/field/test_biotsavart.py`` Taylor tests."""
    curve = CurveXYZFourier(200, 3)
    coeffs = curve.dofs_matrix
    coeffs[1][0] = 1.0
    coeffs[1][1] = 0.5
    coeffs[2][2] = 0.5
    curve.set_dofs(np.concatenate(coeffs))
    rng = np.random.default_rng(1)
    points = np.asarray(17 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
    points += 0.001 * (rng.random(points.shape) - 0.5)
    return Coil(curve, Current(1e4)), points


def _cart_points_to_cyl(points):
    return np.ascontiguousarray(
        np.stack(
            (
                np.sqrt(points[:, 0] * points[:, 0] + points[:, 1] * points[:, 1]),
                np.arctan2(points[:, 1], points[:, 0]),
                points[:, 2],
            ),
            axis=1,
        )
    )


def _assert_cylindrical_points_match_cpu(jax_field, cpu_field):
    np.testing.assert_allclose(
        np.asarray(jax_field.get_points_cyl()),
        np.asarray(cpu_field.get_points_cyl()),
        rtol=0.0,
        atol=1.0e-15,
    )


def _assert_cylindrical_accessors_match_cpu(jax_field, cpu_field):
    _assert_cylindrical_points_match_cpu(jax_field, cpu_field)
    np.testing.assert_allclose(
        np.asarray(jax_field.AbsB()),
        np.asarray(cpu_field.AbsB()),
        rtol=_DIRECT_KERNEL_TOLS["rtol"],
        atol=_DIRECT_KERNEL_TOLS["atol"],
    )
    assert np.asarray(jax_field.AbsB()).shape == np.asarray(cpu_field.AbsB()).shape
    for method_name in ("B_cyl", "A_cyl"):
        np.testing.assert_allclose(
            np.asarray(getattr(jax_field, method_name)()),
            np.asarray(getattr(cpu_field, method_name)()),
            rtol=_DIRECT_KERNEL_TOLS["rtol"],
            atol=_DIRECT_KERNEL_TOLS["atol"],
        )
    np.testing.assert_allclose(
        np.asarray(jax_field.GradAbsB_cyl()),
        np.asarray(cpu_field.GradAbsB_cyl()),
        rtol=_DERIVATIVE_HEAVY_TOLS["first_derivative_rtol"],
        atol=_DERIVATIVE_HEAVY_TOLS["first_derivative_atol"],
    )


# Curves and currents of every supported type, at two points near the coils.

_POINTS = np.array([[0.8, 0.1, 0.2], [1.1, -0.2, -0.1]])
_FIELDS = ("B", "A", "dB_by_dX", "dA_by_dX")
_COMPOSED_WRAPPERS = (
    "shared_filament",
    "centroid_filament",
    "nested_perturbed",
    "perturbed_filament",
    "perturbed_shared_filament",
)


def _curves():
    return create_equally_spaced_curves(
        2, 1, stellsym=False, R0=1.0, R1=0.25, order=2, numquadpoints=32
    )


def _vjp(field, quantity, cotangent) -> Derivative:
    # Warm native caches, including the unit-current A cache.
    getattr(field, quantity)()
    if quantity == "dB_by_dX":
        return field.B_and_dB_vjp(np.zeros_like(_POINTS), cotangent)[1]
    if quantity == "dA_by_dX":
        return field.A_and_dA_vjp(np.zeros_like(_POINTS), cotangent)[1]
    return getattr(field, quantity + "_vjp")(cotangent)


def _gradient(derivative: Derivative, owner: Optimizable) -> np.ndarray:
    return cast(np.ndarray, derivative(owner))


def _cotangent(quantity):
    shape = (2, 3, 3) if "by_dX" in quantity else (2, 3)
    return np.linspace(-0.7, 0.9, np.prod(shape)).reshape(shape)


def _composed_native_field(wrapper, fixed):
    """Build the shared native oracle for composed-curve finite differences and Taylor tests."""
    curve = create_equally_spaced_curves(
        1,
        1,
        stellsym=False,
        R0=1.0,
        R1=0.25,
        order=1,
        numquadpoints=24,
    )[0]
    current = Current(1e5)
    shared = "shared" in wrapper
    rotation = FrameRotation(
        curve.quadpoints,
        order=4 if shared else 1,
        dofs=curve.dofs if shared else None,
    )
    if not shared:
        rotation.local_full_x = np.array([0.2, -0.1, 0.05])
    if fixed:
        curve.fix(0)
        rotation.fix(rotation.local_dof_names[1])
        current.fix_all()
    sampler = GaussianSampler(curve.quadpoints, 1e-3, 0.2, n_derivs=1)
    sample = PerturbationSample(sampler, randomgen=np.random.default_rng(20261005))
    if wrapper == "nested_perturbed":
        wrapped = CurvePerturbed(CurvePerturbed(curve, sample), sample)
    else:
        frame = FramedCurveCentroid if "centroid" in wrapper else FramedCurveFrenet
        wrapped = CurveFilament(frame(curve, rotation), 0.01, 0.02)
        if wrapper.startswith("perturbed"):
            wrapped = CurvePerturbed(wrapped, sample)
    coils = [Coil(wrapped, current)]
    native = BiotSavart(coils).set_points(_POINTS)
    return native, curve, rotation


def _filament(frame, quadpoints):
    """Build a non-circular rotated filament on ``frame`` sampled at ``quadpoints``."""
    reference = create_equally_spaced_curves(
        1, 1, stellsym=False, R0=1.0, R1=0.25, order=2, numquadpoints=24
    )[0]
    curve = CurveXYZFourier(quadpoints, 2)
    curve.x = reference.x + np.random.default_rng(20261011).normal(
        0.0, 0.03, reference.x.size
    )
    rotation = FrameRotation(quadpoints, 1)
    rotation.x = np.array([0.2, -0.1, 0.05])
    return CurveFilament(frame(curve, rotation), 0.05, 0.08)


def _jax_filament_geometry(frame, quadpoints):
    """Evaluate JAX filament ``gamma`` and ``gammadash`` at ``quadpoints``."""
    spec = curve_spec_from_adapter_curve(_filament(frame, quadpoints))
    gamma, gammadash = curve_gamma_and_gammadash_from_spec(spec)
    return np.asarray(gamma), np.asarray(gammadash)


# Kernel tilings: the fused B and dB/dX kernel against autodiff.

# About 45 ulp. The measured worst case on these configurations is about
# 2e-15 at the integrand and 1.4e-15 at the kernel.
_ROUND_OFF = 1e-14


# Kernel tilings: quadrature blocks and coil chunks.

_BLOCK_COIL_COUNT = 5
_QUADRATURE_COUNT = 24
_POINT_COUNT = 11

# (name, coil_chunk_size, quadrature_block_size, loop trip counts) for 5 coils
# of 24 nodes. One or two blocks or chunks take the single or exact two-way
# path, which has no loop; three or more loop over the full blocks or chunks
# and add the tail. coil_chunk_tail loops over two full 2-coil chunks, and the
# quadrature sum of each chunk, traced once in the loop body and once for the
# 1-coil tail, loops over three full 7-node blocks.
_TILINGS = (
    ("two_quadrature_blocks", 0, 16, ()),
    ("three_quadrature_blocks_and_tail", 0, 7, (3,)),
    ("two_coil_chunks", 3, 16, ()),
    ("coil_chunk_tail", 2, 7, (2, 3, 3)),
)

_FORWARD_KERNELS = (
    ("B", "VALUE"),
    ("B", "JACOBIAN"),
    ("A", "VALUE"),
    ("A", "JACOBIAN"),
    ("B", "HESSIAN"),
    ("A", "HESSIAN"),
    ("B", "VALUE_AND_JACOBIAN"),
)

# Measured over every row below (JAX 0.12.0.1), |tiled - untiled| is at most
# 4.5e-16 * (|untiled| + max|untiled|) on CPU and 5.5e-16 * (...) on an RTX
# 5090, two to two and a half float64 epsilons (2.2e-16). Without the
# max|untiled| floor the elementwise relative difference reaches 3.6e-12, on
# entries that cancel to near zero. 1e-14 (45 epsilons) leaves room for other
# summation orders and FMA contraction. One node more or less changes each
# forward kernel's result by at least 1.7% of its largest entry.
_BLOCK_PARITY_RTOL = 1e-14

# Random coil nodes with every coordinate 0.5 to 1.5 in magnitude, so each node
# is at least 0.2 per coordinate away from the points, which lie in [-0.3, 0.3]^3.
_RNG = np.random.default_rng(20261105)
_COIL_SHAPE = (_BLOCK_COIL_COUNT, _QUADRATURE_COUNT, 3)
_GAMMAS = _RNG.uniform(0.5, 1.5, _COIL_SHAPE) * _RNG.choice((-1.0, 1.0), _COIL_SHAPE)
_GAMMADASHS = _RNG.standard_normal(_COIL_SHAPE)
_CURRENTS = _RNG.uniform(0.5, 1.5, _BLOCK_COIL_COUNT) * 1e4
_BLOCK_POINTS = _RNG.uniform(-0.3, 0.3, (_POINT_COUNT, 3))
_BLOCK_COTANGENT = _RNG.standard_normal((_POINT_COUNT, 3))
_FORWARD_INPUTS = (_BLOCK_POINTS, _GAMMAS, _GAMMADASHS, _CURRENTS)
_VJP_INPUTS = (_BLOCK_POINTS, _BLOCK_COTANGENT, _GAMMAS, _GAMMADASHS, _CURRENTS)


def _forward_kernel(integrand, diff_mode, coil_chunk_size, quadrature_block_size):
    return core_biotsavart._make_kernel(
        core_biotsavart._Integrand[integrand],
        core_biotsavart._DiffMode[diff_mode],
        coil_chunk_size,
        quadrature_block_size,
        0,
    )


def _loop_trip_counts(jaxpr):
    """Sorted trip counts of the scans in ``jaxpr`` and the jaxprs nested in it.

    ``lax.fori_loop`` with static bounds traces to a scan of that length.
    """
    return tuple(
        sorted(
            [eqn.params["length"] for eqn in jaxpr.eqns if eqn.primitive.name == "scan"]
            + [
                count
                for eqn in jaxpr.eqns
                for sub_jaxpr in jaxprs_in_params(eqn.params)
                for count in _loop_trip_counts(sub_jaxpr)
            ]
        )
    )


# The helpers below return NumPy values, so a test row holds no JAX object when
# its case cleans up the runtime.
def _loops_and_result(kernel, inputs):
    """Trace ``kernel`` once; return its loop trip counts and its compiled result."""
    traced = kernel.trace(*inputs)
    result = traced.lower().compile()(*inputs)
    return _loop_trip_counts(traced.jaxpr.jaxpr), jax.tree.map(np.asarray, result)


def _B_vjp(coil_chunk_size, quadrature_block_size):
    kernel = core_biotsavart._make_B_vjp_kernel(
        coil_chunk_size, quadrature_block_size, 0
    )
    return jax.tree.map(np.asarray, kernel(*_VJP_INPUTS))


def _assert_results_match(tiled, untiled):
    """Elementwise rtol with an absolute floor at rtol times the largest value."""
    jax.tree.map(
        lambda tiled_leaf, untiled_leaf: np.testing.assert_allclose(
            tiled_leaf,
            untiled_leaf,
            rtol=_BLOCK_PARITY_RTOL,
            atol=_BLOCK_PARITY_RTOL * np.max(np.abs(untiled_leaf)),
        ),
        tiled,
        untiled,
    )


# Kernel tilings: Hessian point chunks.

def _coil_set(rng):
    nquad = 12
    phi = np.linspace(0.0, 2.0 * np.pi, nquad, endpoint=False)
    gammas = []
    gammadashs = []
    for center in ((0.0, 0.0, 0.0), (0.3, 0.1, -0.2)):
        gamma = np.stack(
            (
                np.cos(phi) + center[0],
                np.sin(phi) + center[1],
                np.zeros(nquad) + center[2],
            ),
            axis=-1,
        )
        gammadash = np.stack((-np.sin(phi), np.cos(phi), np.zeros(nquad)), axis=-1)
        gammas.append(gamma)
        gammadashs.append(gammadash)
    currents = rng.uniform(0.5, 1.5, size=2)
    return (
        jnp.asarray(np.stack(gammas)),
        jnp.asarray(np.stack(gammadashs)),
        jnp.asarray(currents),
    )


# Compiled-program form of the per-coil reductions.

# Sizes are (coils, nodes, points). Five coils is the only length-5 axis: 24
# nodes (padded to 32 by the pairwise sum), 12-node blocks (padded to 16),
# 7-point chunks with a 6-point tail of 20 points, and 3 Cartesian components.
_LAYOUT_COIL_COUNT = 5
_SIZES = (_LAYOUT_COIL_COUNT, 24, 20)
_POINT_CHUNK_SIZE = 7
# (coil_chunk_size, quadrature_block_size, point_chunk_size): unblocked, and
# two quadrature blocks as in the fast-lane tuning (128 nodes in 64-node blocks).
_TUNINGS = ((0, 0, _POINT_CHUNK_SIZE), (0, 12, _POINT_CHUNK_SIZE))
# The fast-lane tiling at which XLA formed the A100 concatenate fusions: one
# 1024-point chunk and 128 nodes in two 64-node blocks; 31 coils is the only
# length-31 axis.
_FAST_LANE_COIL_COUNT = 31
_FAST_LANE_SIZES = (_FAST_LANE_COIL_COUNT, 128, 1024)
_FAST_LANE_TUNING = (0, 64, 1024)

# Forward kernel name -> ``_DiffMode`` member name.
_FORWARD_DIFF_MODES = {"B": "VALUE", "dB_by_dX": "JACOBIAN"}

_COMPUTATION = re.compile(r"^(?:ENTRY\s+)?%([\w.\-]+)\s.*\{\s*$")
_INSTRUCTION = re.compile(
    r"^\s*(?:ROOT\s+)?%(?P<name>[\w.\-]+)\s*=\s*"
    r"(?P<shape>\([^=]*?\)|\S+)\s+(?P<opcode>[\w\-]+)\((?P<rest>.*)$"
)
_ARRAY_SHAPE = re.compile(r"^[a-z]\w*\[(?P<dims>[\d,]*)\](?:\{(?P<layout>[\d,]*))?")
_REDUCED_DIMENSIONS = re.compile(r"dimensions=\{([\d,]*)\}")


@dataclass(frozen=True)
class _Instruction:
    name: str
    shape: str
    opcode: str
    operands: tuple[str, ...]
    attributes: str


def _parse_hlo(text):
    """Map each computation of an HLO module's text to its instructions."""
    computations: dict[str, list[_Instruction]] = {}
    current: list[_Instruction] = []
    for line in text.splitlines():
        computation = _COMPUTATION.match(line)
        if computation:
            current = computations.setdefault(computation.group(1), [])
            continue
        instruction = _INSTRUCTION.match(line)
        if instruction is None:
            continue
        rest = instruction.group("rest")
        operand_text, _, attributes = rest.partition(")")
        current.append(
            _Instruction(
                name=instruction.group("name"),
                shape=instruction.group("shape"),
                opcode=instruction.group("opcode"),
                operands=tuple(re.findall(r"%([\w.\-]+)", operand_text)),
                attributes=attributes,
            )
        )
    return computations


def _per_coil_reduces(computations, coil_count):
    """Map each per-coil reduce of the module to its operand's shape and layout.

    A per-coil reduce keeps the coil axis: it sums a per-coil-node array over
    quadrature nodes, or over the point chunk in the VJP. Reductions over the
    coil axis (the current contraction) are not per-coil. HLO instruction names
    are unique within a module. Values: (operand shape, dims, minor axis,
    reduced axes).
    """
    instructions = [inst for body in computations.values() for inst in body]
    shapes = {instruction.name: instruction.shape for instruction in instructions}
    found = {}
    for instruction in instructions:
        if instruction.opcode != "reduce":
            continue
        shape = shapes[instruction.operands[0]]
        ((dims_text, layout),) = _ARRAY_SHAPE.findall(shape)
        dims = tuple(int(size) for size in dims_text.split(",") if size)
        minor = int(layout.split(",")[0]) if layout else len(dims) - 1
        (axes,) = _REDUCED_DIMENSIONS.findall(instruction.attributes)
        reduced = {int(axis) for axis in axes.split(",") if axis}
        if any(
            size == coil_count and axis not in reduced for axis, size in enumerate(dims)
        ):
            found[instruction.name] = (shape, dims, minor, reduced)
    return found


def _kernel_and_arguments(kernel_name, tuning, sizes):
    """Return one kernel and float64 shape structs of ``sizes`` for its inputs."""
    coils, nodes, point_count = sizes

    def spec(*shape):
        return jax.ShapeDtypeStruct(shape, jnp.float64)

    points, coil_data, currents = (
        spec(point_count, 3),
        spec(coils, nodes, 3),
        spec(coils),
    )
    if kernel_name == "B_vjp":
        kernel = core_biotsavart._make_B_vjp_kernel(*tuning)
        return kernel, (points, points, coil_data, coil_data, currents)
    kernel = core_biotsavart._make_kernel(
        core_biotsavart._Integrand.B,
        core_biotsavart._DiffMode[_FORWARD_DIFF_MODES[kernel_name]],
        *tuning,
    )
    return kernel, (points, coil_data, coil_data, currents)


def _compiled_hlo(kernel_name, tuning, sizes):
    """Compile one kernel for JAX's default device and return its HLO."""
    kernel, arguments = _kernel_and_arguments(kernel_name, tuning, sizes)
    return _parse_hlo(kernel.lower(*arguments).compile().as_text())


def _per_coil_quadrature_sums(jaxpr, coil_count):
    """Yield ``(sum, reads)`` for each per-coil quadrature sum in ``jaxpr`` and
    the jaxprs nested in it.

    ``pairwise_sum_axis`` ends in a ``squeeze`` of a ``reduce_sum``; its result
    is a per-coil sum when it keeps the coil axis. ``reads`` holds the
    equations that read the sum, and None if its jaxpr returns it.
    """
    producers = {var: eqn for eqn in jaxpr.eqns for var in eqn.outvars}
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "squeeze" and coil_count in eqn.outvars[0].aval.shape:
            source = producers.get(eqn.invars[0])
            if source is not None and source.primitive.name == "reduce_sum":
                (total,) = eqn.outvars
                reads = [
                    reader
                    for reader in jaxpr.eqns
                    if any(operand is total for operand in reader.invars)
                ]
                returned = any(result is total for result in jaxpr.outvars)
                yield total, reads + [None] * returned
        for sub_jaxpr in jaxprs_in_params(eqn.params):
            yield from _per_coil_quadrature_sums(sub_jaxpr, coil_count)


def _is_contraction(eqn):
    """Whether ``eqn`` contracts a per-coil sum with a vector: the currents, or
    in the VJP the cotangent of that contraction."""
    if eqn is None or eqn.primitive.name != "dot_general":
        return False
    return sorted(len(var.aval.shape) for var in eqn.invars)[0] == 1


# Coil-DOF extraction maps.

_NQUAD = 8
_ORDER = 1
_SPLIT_CURVE_FIXED = ("yc(0)", "zc(1)")


def _bits(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).view(np.uint64)


def coil_graph():
    """Two coils: the first curve has two interior dofs fixed, the second current is fixed."""
    curves = [CurveXYZFourier(_NQUAD, _ORDER) for _ in range(2)]
    for index, curve in enumerate(curves):
        curve.x = (
            0.1 * (index + 1) * np.arange(1, int(curve.dof_size) + 1, dtype=np.float64)
        )
    for name in _SPLIT_CURVE_FIXED:
        curves[0].fix(name)
    currents = [Current(1.0e4), Current(2.0e4)]
    currents[1].fix_all()
    coils = [Coil(curve, current) for curve, current in zip(curves, currents)]
    bs_jax = JaxBiotSavart(coils)
    oracle = BiotSavart(coils)
    assert list(oracle.dof_names) == list(bs_jax.dof_names)
    return curves, currents, bs_jax, oracle


def _owner_start(bs_jax, opt) -> int:
    return list(bs_jax.dof_names).index(f"{opt.name}:{opt.local_dof_names[0]}")


def _lowered_extraction_text(bs_jax) -> str:
    spec = bs_jax.coil_dof_extraction_spec()
    free_dofs = jnp.asarray(np.linspace(-1.0, 1.0, bs_jax.dof_size))
    lowered = jit(lambda dofs: coil_specs_from_dof_extraction_spec(spec, dofs))
    return lowered.lower(free_dofs).as_text()


class TestBiotSavartJaxPhysics(JaxTestCase):
    """Field properties that hold independently of the native implementation."""

    def test_circular_loop_on_axis_field_is_analytic(self):
        """On the axis of a circular loop of radius R carrying I,
        B = (0, 0, mu0 I R^2 / (2 (R^2 + z^2)^(3/2))).

        The integrand is constant along the loop for points on its axis, so the
        quadrature is exact and only round-off remains.
        """
        radius, current = 1.3, 2.0e5
        curve = CurveXYZFourier(64, 1)
        curve.x = np.array(
            [radius if name in ("xc(1)", "ys(1)") else 0.0 for name in curve.local_dof_names]
        )
        z = np.array([-2.0, -0.4, 0.0, 0.25, 3.0])
        points = np.stack((np.zeros_like(z), np.zeros_like(z), z), axis=1)
        field = JaxBiotSavart([Coil(curve, Current(current))]).set_points(points)
        mu0 = 4.0e-7 * np.pi
        expected = np.zeros_like(points)
        expected[:, 2] = mu0 * current * radius**2 / (2.0 * (radius**2 + z**2) ** 1.5)
        np.testing.assert_allclose(
            np.asarray(field.B()), expected, rtol=1e-14, atol=1e-14 * expected[:, 2].max()
        )

    def test_field_is_divergence_free(self):
        """The trace of dB/dX vanishes to round-off of its largest entry at each
        point, around the NCSX magnetic axis."""
        bs, points_np, _, _, _ = _ncsx_biotsavart_parity_fixture()
        field = JaxBiotSavart(list(bs._coils)).set_points(points_np)
        dB = np.asarray(field.dB_by_dX())
        divergence = np.trace(dB, axis1=1, axis2=2)
        self.assertLess(
            np.max(np.abs(divergence) / np.max(np.abs(dB), axis=(1, 2))), 1e-13
        )


class TestBiotSavartJaxNativeParity(JaxTestCase):
    """Kernels and public accessors against native BiotSavart on the NCSX coils."""

    def test_kernel_parity_ncsx(self):
        """Raw JAX kernels match native ``BiotSavart`` on the NCSX parity fixture."""
        direct = (_DIRECT_KERNEL_TOLS["rtol"], _DIRECT_KERNEL_TOLS["atol"])
        first_derivative = (
            _DERIVATIVE_HEAVY_TOLS["first_derivative_rtol"],
            _DERIVATIVE_HEAVY_TOLS["first_derivative_atol"],
        )
        for quantity, kernel, (rtol, atol) in (
            ("B", biot_savart_B, direct),
            ("A", biot_savart_A, direct),
            ("dB_by_dX", biot_savart_dB_by_dX, first_derivative),
            ("dA_by_dX", biot_savart_dA_by_dX, first_derivative),
        ):
            with self.subTest(quantity=quantity), self.case():
                self._case_kernel_parity_ncsx(quantity, kernel, rtol, atol)

    def _case_kernel_parity_ncsx(self, quantity, kernel, rtol, atol):
        """Run one row with case-local objects released before runtime cleanup."""
        bs, points_np, gammas_np, gds_np, currents_np = (
            _ncsx_biotsavart_parity_fixture()
        )
        expected = getattr(bs, quantity)()

        actual = kernel(
            jnp.array(points_np),
            jnp.array(gammas_np),
            jnp.array(gds_np),
            jnp.array(currents_np),
        )

        np.testing.assert_allclose(np.array(actual), expected, rtol=rtol, atol=atol)

    def test_cylindrical_public_accessors_parity_ncsx(self):
        """``JaxBiotSavart`` owns cylindrical public accessors on its boundary."""

        bs, points_np, _, _, _ = _ncsx_biotsavart_parity_fixture()
        points_cyl = _cart_points_to_cyl(points_np)
        points_cyl[:, 1] += 2.0 * np.pi
        bs.set_points_cyl(points_cyl)

        bs_jax = JaxBiotSavart(list(bs._coils))
        self.assertTrue(
            bs_jax.set_points_cyl(points_cyl) is bs_jax,
            "bs_jax.set_points_cyl(points_cyl) is bs_jax",
        )
        self.assertTrue(
            bs_jax.set_points_cart(points_np) is bs_jax,
            "bs_jax.set_points_cart(points_np) is bs_jax",
        )
        self.assertTrue(
            bs_jax.set_points_cyl(points_cyl) is bs_jax,
            "bs_jax.set_points_cyl(points_cyl) is bs_jax",
        )

        _assert_cylindrical_accessors_match_cpu(bs_jax, bs)

    def test_cylindrical_public_accessors_use_cached_phi_basis_ncsx(self):
        """``B_cyl`` / ``A_cyl`` / ``GradAbsB_cyl`` use cached cylindrical phi."""

        bs, _, _, _, _ = _ncsx_biotsavart_parity_fixture()
        points_cyl = np.ascontiguousarray(
            np.array(
                [
                    [0.0, 1.234, 0.1],
                    [0.0, -1.5, -0.2],
                    [0.7, 2.0 * np.pi + 0.4, 0.3],
                ],
                dtype=np.float64,
            )
        )
        bs.set_points_cyl(points_cyl)

        bs_jax = JaxBiotSavart(list(bs._coils))
        bs_jax.set_points_cyl(points_cyl)

        _assert_cylindrical_accessors_match_cpu(bs_jax, bs)

    def test_cartesian_public_accessors_normalize_cylindrical_phi_ncsx(self):
        """Cartesian point conversion follows C++ ``get_points_cyl_impl``."""

        bs, _, _, _, _ = _ncsx_biotsavart_parity_fixture()
        points_cart = np.ascontiguousarray(
            np.array(
                [
                    [0.5, -0.5, 0.1],
                    [-0.5, -0.25, -0.2],
                    [0.0, 0.0, 0.3],
                ],
                dtype=np.float64,
            )
        )
        bs.set_points_cart(points_cart)

        bs_jax = JaxBiotSavart(list(bs._coils))
        bs_jax.set_points_cart(points_cart)

        _assert_cylindrical_points_match_cpu(bs_jax, bs)

    def test_nonfinite_point_leaves_other_cylindrical_angles_native(self):
        """A nonfinite point does not change the cylindrical angle of finite points,
        for Cartesian input (wrap to [0, 2*pi)) and cylindrical input (fmod)."""
        bs, _, _, _, _ = _ncsx_biotsavart_parity_fixture()
        bs_jax = JaxBiotSavart(list(bs._coils))
        for setter, points in (
            ("set_points_cart", [[1.0, -1.0, 0.0], [np.inf, 0.0, 0.0]]),
            ("set_points_cyl", [[1.0, 7.0, 0.0], [1.0, np.nan, 0.0]]),
        ):
            with self.subTest(setter=setter):
                points = np.ascontiguousarray(points, dtype=np.float64)
                getattr(bs, setter)(points)
                getattr(bs_jax, setter)(points)
                np.testing.assert_array_equal(
                    bs_jax.get_points_cyl()[0], bs.get_points_cyl()[0]
                )

    def test_default_evaluation_point_is_origin_like_native(self):
        """Without set_points the field is evaluated at one point at the origin."""
        bs, _, _, _, _ = _ncsx_biotsavart_parity_fixture()
        native = BiotSavart(list(bs._coils))
        bs_jax = JaxBiotSavart(list(bs._coils))
        np.testing.assert_array_equal(bs_jax.get_points_cart(), np.zeros((1, 3)))
        np.testing.assert_allclose(
            np.asarray(bs_jax.B()),
            native.B(),
            rtol=_DIRECT_KERNEL_TOLS["rtol"],
            atol=_DIRECT_KERNEL_TOLS["atol"],
        )

    def test_empty_coil_set_GradAbsB_is_nan_like_native(self):
        """With no coils |B| is identically zero and its gradient is native's 0/0."""
        _, points_np, _, _, _ = _ncsx_biotsavart_parity_fixture()
        native = BiotSavart([]).set_points(points_np)
        bs_jax = JaxBiotSavart([]).set_points(points_np)
        np.testing.assert_array_equal(np.asarray(bs_jax.B()), native.B())
        self.assertTrue(np.all(np.isnan(native.GradAbsB())))
        np.testing.assert_array_equal(np.asarray(bs_jax.GradAbsB()), native.GradAbsB())
        # The NaN is built on device, so it also holds under the strict guard.
        bs_jax.set_points(points_np)
        with disallow_host_transfers():
            grad_abs_B = bs_jax.GradAbsB()
        np.testing.assert_array_equal(np.asarray(grad_abs_B), native.GradAbsB())

    def test_per_current_derivatives_run_under_strict_transfer_guard(self):
        """Every per-coil unit-current accessor builds its unit current and its
        per-coil rows on device, so it runs under the strict guard at native
        values: the public accessors, and their builder unchunked and in coil chunks."""
        for chunk_size in (None, 0, 7):
            with self.subTest(chunk_size=chunk_size), self.case():
                self._case_per_current_derivatives_run_under_strict_transfer_guard(chunk_size)

    def _case_per_current_derivatives_run_under_strict_transfer_guard(self, chunk_size):
        """Run one row with case-local objects released before runtime cleanup.

        ``chunk_size`` None calls the public accessors; an integer calls their
        builder with that coil batch size.
        """
        bs, points_np, _, _, _ = _ncsx_biotsavart_parity_fixture()
        bs_jax = JaxBiotSavart(list(bs._coils))
        bs_jax.set_points(points_np)
        points, coil_set_spec = bs_jax.get_points_cart_ref(), bs_jax.coil_set_spec()
        direct = (_DIRECT_KERNEL_TOLS["rtol"], _DIRECT_KERNEL_TOLS["atol"])
        first = (
            _DERIVATIVE_HEAVY_TOLS["first_derivative_rtol"],
            _DERIVATIVE_HEAVY_TOLS["first_derivative_atol"],
        )
        second = (
            _DERIVATIVE_HEAVY_TOLS["second_derivative_rtol"],
            _DERIVATIVE_HEAVY_TOLS["second_derivative_atol"],
        )
        for cache_method, list_method, kernel, tols in (
            ("B", "dB_by_dcoilcurrents", biot_savart_B, direct),
            ("dB_by_dX", "d2B_by_dXdcoilcurrents", biot_savart_dB_by_dX, first),
            ("d2B_by_dXdX", "d3B_by_dXdXdcoilcurrents", core_biotsavart.biot_savart_d2B_by_dXdX, second),
            ("A", "dA_by_dcoilcurrents", biot_savart_A, direct),
            ("dA_by_dX", "d2A_by_dXdcoilcurrents", biot_savart_dA_by_dX, first),
            ("d2A_by_dXdX", "d3A_by_dXdXdcoilcurrents", core_biotsavart.biot_savart_d2A_by_dXdX, second),
        ):
            getattr(bs, cache_method)()
            with disallow_host_transfers():
                if chunk_size is None:
                    jax_list = getattr(bs_jax, list_method)()
                else:
                    jax_list = _per_coil_unit_field_with_batch_size(
                        points, coil_set_spec, kernel, batch_size=chunk_size
                    )
            cpu_list = getattr(bs, list_method)()
            self.assertEqual(len(jax_list), len(cpu_list))
            for k, (j_entry, c_entry) in enumerate(zip(jax_list, cpu_list)):
                np.testing.assert_allclose(
                    np.asarray(j_entry), c_entry, rtol=tols[0], atol=tols[1],
                    err_msg=f"{list_method} coil {k}",
                )

    def test_B_vjp_parity_ncsx(self):
        """Native Biot-Savart parity on identical coils, points and cotangents."""

        bs_cpu, points_np, _, _, _ = _ncsx_biotsavart_parity_fixture()
        coils = list(bs_cpu._coils)

        bs_jax = JaxBiotSavart(coils)
        bs_jax.set_points(points_np)

        v = np.asarray(bs_cpu.B(), dtype=np.float64).copy()
        deriv_cpu = cast(Derivative, bs_cpu.B_vjp(v))
        deriv_jax = bs_jax.B_vjp(v)

        for coil in coils:
            np.testing.assert_allclose(
                np.asarray(deriv_jax(coil)),
                np.asarray(deriv_cpu(coil)),
                rtol=_DERIVATIVE_HEAVY_TOLS["first_derivative_rtol"],
                atol=_DERIVATIVE_HEAVY_TOLS["first_derivative_atol"],
                err_msg=(
                    "JaxBiotSavart.B_vjp() does not match BiotSavart.B_vjp() "
                    "on the NCSX parity fixture"
                ),
            )

    def test_coil_chunked_kernels_and_pullback_match_native_ncsx(self):
        """Coil chunking leaves B, dB/dX and the B pullback at native values.

        The 18 NCSX coils form two chunks of 10 and 8 (the exact two-chunk
        path), or two chunks of 7 and a 4-coil tail (the padded loop).
        """
        for chunk_size in (10, 7):
            with self.subTest(chunk_size=chunk_size), self.case():
                self._case_coil_chunked_kernels_and_pullback_match_native_ncsx(
                    chunk_size
                )

    def _case_coil_chunked_kernels_and_pullback_match_native_ncsx(self, chunk_size):
        """Run one row with case-local objects released before runtime cleanup.

        The kernels come from the factories at ``chunk_size`` coils per chunk;
        the chunked pullback is compared with the public one, which
        ``test_B_vjp_parity_ncsx`` compares with native.
        """
        bs, points_np, gammas_np, gds_np, currents_np = (
            _ncsx_biotsavart_parity_fixture()
        )
        self.assertEqual(len(currents_np), 18)
        B_kernel, dB_kernel = (
            core_biotsavart._make_kernel(
                core_biotsavart._Integrand.B, diff_mode, chunk_size, 0, 0
            )
            for diff_mode in (core_biotsavart._DiffMode.VALUE, core_biotsavart._DiffMode.JACOBIAN)
        )
        np.testing.assert_allclose(
            np.asarray(B_kernel(points_np, gammas_np, gds_np, currents_np)),
            bs.B(),
            rtol=_DIRECT_KERNEL_TOLS["rtol"],
            atol=_DIRECT_KERNEL_TOLS["atol"],
        )
        np.testing.assert_allclose(
            np.asarray(dB_kernel(points_np, gammas_np, gds_np, currents_np)),
            bs.dB_by_dX(),
            rtol=_DERIVATIVE_HEAVY_TOLS["first_derivative_rtol"],
            atol=_DERIVATIVE_HEAVY_TOLS["first_derivative_atol"],
        )
        v = np.asarray(bs.B(), dtype=np.float64).copy()
        chunked = core_biotsavart._make_B_vjp_kernel(chunk_size, 0, 0)(
            points_np, v, gammas_np, gds_np, currents_np
        )
        public = core_biotsavart.biot_savart_B_vjp(points_np, v, gammas_np, gds_np, currents_np)
        for actual, expected in zip(chunked, public, strict=True):
            np.testing.assert_allclose(
                np.asarray(actual),
                np.asarray(expected),
                rtol=_DERIVATIVE_HEAVY_TOLS["first_derivative_rtol"],
                atol=_DERIVATIVE_HEAVY_TOLS["first_derivative_atol"],
            )

    def test_adapter_derivative_parity_ncsx(self):
        """``JaxBiotSavart`` spatial derivatives match native ``BiotSavart`` on the NCSX
        parity fixture."""
        first_derivative = (
            _DERIVATIVE_HEAVY_TOLS["first_derivative_rtol"],
            _DERIVATIVE_HEAVY_TOLS["first_derivative_atol"],
        )
        second_derivative = (
            _DERIVATIVE_HEAVY_TOLS["second_derivative_rtol"],
            _DERIVATIVE_HEAVY_TOLS["second_derivative_atol"],
        )
        for quantity, (rtol, atol) in (
            ("dA_by_dX", first_derivative),
            ("d2B_by_dXdX", second_derivative),
            ("d2A_by_dXdX", second_derivative),
        ):
            with self.subTest(quantity=quantity), self.case():
                self._case_adapter_derivative_parity_ncsx(quantity, rtol, atol)

    def _case_adapter_derivative_parity_ncsx(self, quantity, rtol, atol):
        """Run one row with case-local objects released before runtime cleanup."""
        bs, points_np, _, _, _ = _ncsx_biotsavart_parity_fixture()
        expected = getattr(bs, quantity)()

        bs_jax = JaxBiotSavart(list(bs._coils))
        bs_jax.set_points(points_np)
        actual = getattr(bs_jax, quantity)()

        np.testing.assert_allclose(np.array(actual), expected, rtol=rtol, atol=atol)

    def test_GradAbsB_taylor_ncsx(self):
        """GradAbsB matches central differences of AbsB with second-order convergence
        under shrinking point steps."""

        bs, points_np, _, _, _ = _ncsx_biotsavart_parity_fixture()
        bs_jax = JaxBiotSavart(list(bs._coils))
        bs_jax.set_points(points_np)
        grad = np.asarray(bs_jax.GradAbsB())
        for direction in np.eye(3):
            with self.subTest(direction=direction):
                deriv = grad @ direction
                err = 1e6
                for i in range(5):
                    eps = 1e-3 * 0.5**i
                    bs_jax.set_points(points_np + eps * direction)
                    plus = np.asarray(bs_jax.AbsB())[:, 0]
                    bs_jax.set_points(points_np - eps * direction)
                    minus = np.asarray(bs_jax.AbsB())[:, 0]
                    new_err = np.linalg.norm((plus - minus) / (2 * eps) - deriv)
                    self.assertLess(new_err, 0.3 * err)
                    err = new_err

    def test_per_coil_unit_field_vectorizes_within_quadrature_group(self):
        """Unit-current fields vectorize per quadrature group and return values in
        public coil order."""
        points = jnp.asarray([[0.0, 0.0, 0.0], [0.25, -0.5, 1.0]], dtype=jnp.float64)
        group0_gammas = jnp.arange(18, dtype=jnp.float64).reshape(2, 3, 3)
        group0_gammadashs = group0_gammas + 0.5
        group1_gammas = jnp.arange(9, dtype=jnp.float64).reshape(1, 3, 3) - 2.0
        group1_gammadashs = group1_gammas - 0.25
        coil_set_spec = GroupedCoilSetSpec(
            groups=(
                CoilGroupSpec(
                    gammas=group0_gammas,
                    gammadashs=group0_gammadashs,
                    currents=jnp.asarray([3.0, 4.0], dtype=jnp.float64),
                    coil_indices=(2, 0),
                ),
                CoilGroupSpec(
                    gammas=group1_gammas,
                    gammadashs=group1_gammadashs,
                    currents=jnp.asarray([5.0], dtype=jnp.float64),
                    coil_indices=(1,),
                ),
            )
        )
        calls = []

        def kernel(kernel_points, gammas, gammadashs, currents):
            calls.append(gammas.shape)
            value = jnp.sum(gammas) + jnp.sum(gammadashs) + jnp.sum(currents)
            return jnp.broadcast_to(value, kernel_points.shape)

        results = _per_coil_unit_field(points, coil_set_spec, kernel)

        self.assertTrue(len(calls) == 2, "len(calls) == 2")
        self.assertTrue(
            calls == [(1, 3, 3), (1, 3, 3)], "calls == [(1, 3, 3), (1, 3, 3)]"
        )
        self.assertTrue(len(results) == 3, "len(results) == 3")
        np.testing.assert_allclose(
            np.asarray(results[0]),
            np.broadcast_to(
                np.sum(np.asarray(group0_gammas[1]))
                + np.sum(np.asarray(group0_gammadashs[1]))
                + 1.0,
                points.shape,
            ),
        )
        np.testing.assert_allclose(
            np.asarray(results[1]),
            np.broadcast_to(
                np.sum(np.asarray(group1_gammas[0]))
                + np.sum(np.asarray(group1_gammadashs[0]))
                + 1.0,
                points.shape,
            ),
        )
        np.testing.assert_allclose(
            np.asarray(results[2]),
            np.broadcast_to(
                np.sum(np.asarray(group0_gammas[0]))
                + np.sum(np.asarray(group0_gammadashs[0]))
                + 1.0,
                points.shape,
            ),
        )


class TestBiotSavartJaxSpatialTaylor(JaxTestCase):
    """Shrinking-step Taylor tests of the ``JaxBiotSavart`` spatial derivatives.

    Mirrors the native ``tests/field/test_biotsavart.py`` checks on its coil
    and points: steps ``0.5**i`` for ``i`` in ``5..9``, a one-sided difference
    with error ratio 0.55 for first derivatives and a central difference of
    the first derivative with error ratio 0.30 for second derivatives.
    """

    def _assert_spatial_taylor(self, value_name, derivative_name, *, central, ratio):
        """Assert ``derivative[:, k]`` is the shrinking-step limit along ``e_k``."""
        coil, points_np = _native_taylor_fixture()
        bs_jax = JaxBiotSavart([coil])
        bs_jax.set_points(points_np)
        value0 = np.asarray(getattr(bs_jax, value_name)())
        derivative = np.asarray(getattr(bs_jax, derivative_name)())
        for k, direction in enumerate(np.eye(3)):
            with self.subTest(direction=k):
                expected = derivative[:, k]
                err = 1e6
                for i in range(5, 10):
                    eps = 0.5**i
                    bs_jax.set_points(points_np + eps * direction)
                    plus = np.asarray(getattr(bs_jax, value_name)())
                    if central:
                        bs_jax.set_points(points_np - eps * direction)
                        minus = np.asarray(getattr(bs_jax, value_name)())
                        estimate = (plus - minus) / (2.0 * eps)
                    else:
                        estimate = (plus - value0) / eps
                    new_err = np.linalg.norm(estimate - expected)
                    self.assertLess(new_err, ratio * err)
                    err = new_err

    def test_dB_by_dX_taylor(self):
        """dB_by_dX is the first-order shrinking-step limit of B along each axis."""
        self._assert_spatial_taylor("B", "dB_by_dX", central=False, ratio=0.55)

    def test_d2B_by_dXdX_taylor(self):
        """d2B_by_dXdX is the central shrinking-step limit of dB_by_dX along each axis."""
        self._assert_spatial_taylor("dB_by_dX", "d2B_by_dXdX", central=True, ratio=0.30)

    def test_dA_by_dX_taylor(self):
        """dA_by_dX is the first-order shrinking-step limit of A along each axis."""
        self._assert_spatial_taylor("A", "dA_by_dX", central=False, ratio=0.55)

    def test_d2A_by_dXdX_taylor(self):
        """d2A_by_dXdX is the central shrinking-step limit of dA_by_dX along each axis."""
        self._assert_spatial_taylor("dA_by_dX", "d2A_by_dXdX", central=True, ratio=0.30)


class TestBiotSavartJaxCurveAndCurrentTypes(JaxTestCase):
    """Fields and VJPs for framed, filament, perturbed and symmetric curves, affine
    currents, fixed owners and single-precision inputs, against native and
    finite differences.
    """

    def test_filament_gammadash_is_quadpoint_derivative(self):
        """JAX filament gammadash, built from the frame derivative kernels, is the
        quadpoint derivative of gamma and matches native for both frames."""
        for frame in [FramedCurveFrenet, FramedCurveCentroid]:
            with self.subTest(frame=frame.__name__), self.case():
                self._case_filament_gammadash_is_quadpoint_derivative(frame)

    def _case_filament_gammadash_is_quadpoint_derivative(self, frame):
        """Run one row with case-local objects released before runtime cleanup."""
        quadpoints = np.linspace(0.0, 1.0, 24, endpoint=False)
        gamma, gammadash = _jax_filament_geometry(frame, quadpoints)
        error = 1e6
        for exponent in range(5):
            step = 1e-2 * 0.5**exponent
            plus, _ = _jax_filament_geometry(frame, quadpoints + step)
            minus, _ = _jax_filament_geometry(frame, quadpoints - step)
            next_error = np.linalg.norm((plus - minus) / (2 * step) - gammadash)
            self.assertLess(next_error, 0.3 * error)
            error = next_error
        native = _filament(frame, quadpoints)
        np.testing.assert_allclose(gamma, native.gamma(), rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(
            gammadash, native.gammadash(), rtol=1e-12, atol=1e-14
        )

    def test_affine_current_vjps_match_native_and_finite_difference(self):
        """Affine-current fields, owner partials and flat gradients agree with native and
        finite-difference references."""
        for scale in [1.0, -2.5]:
            for quantity in _FIELDS:
                with self.subTest(scale=scale, quantity=quantity), self.case():
                    self._case_affine_current_vjps_match_native_and_finite_difference(
                        scale, quantity
                    )

    def _case_affine_current_vjps_match_native_and_finite_difference(
        self, scale, quantity
    ):
        """Run one row with case-local objects released before runtime cleanup."""
        curves = _curves()
        for curve in curves:
            curve.fix_all()
        first, total = Current(3.0), Current(10.0)
        total.fix_all()
        coils = [
            Coil(curves[0], first),
            Coil(curves[1], scale * (total - first)),
        ]
        native = BiotSavart(coils).set_points(_POINTS)
        adapter = JaxBiotSavart(coils).set_points(_POINTS)
        cotangent = _cotangent(quantity)
        np.testing.assert_allclose(
            getattr(adapter, quantity)(),
            getattr(native, quantity)(),
            rtol=1e-12,
            atol=1e-14,
        )
        expected = _gradient(_vjp(native, quantity, cotangent), native)
        actual = _gradient(_vjp(adapter, quantity, cotangent), adapter)
        value = cast(np.ndarray, first.local_full_x).copy()
        step = 1e-3
        first.local_full_x = value + step
        plus = np.sum(getattr(native, quantity)() * cotangent)
        first.local_full_x = value - step
        minus = np.sum(getattr(native, quantity)() * cotangent)
        first.local_full_x = value
        finite_difference = (plus - minus) / (2 * step)
        pullback = {
            "B": adapter.B_pullback_native,
            "A": adapter.A_pullback_native,
            "dB_by_dX": adapter.dB_by_dX_pullback_native,
            "dA_by_dX": adapter.dA_by_dX_pullback_native,
        }[quantity](cotangent)
        flat = adapter.coil_cotangents_to_dofs_gradient(
            pullback.d_coil_arrays, pullback.coil_indices
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(flat, expected, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(actual, finite_difference, rtol=1e-9, atol=1e-14)
        native_partials = cast(
            Derivative,
            _vjp(native, quantity, cotangent)(native, as_derivative=True),
        )
        adapter_partials = cast(
            Derivative,
            _vjp(adapter, quantity, cotangent)(adapter, as_derivative=True),
        )
        for owner in native_partials.data:
            np.testing.assert_allclose(
                adapter_partials.data[owner],
                native_partials.data[owner],
                rtol=1e-11,
                atol=1e-13,
            )

    def test_affine_current_vjps_shrinking_steps(self):
        """Affine-current VJPs retain linearity through five halved central steps."""
        for scale in (1.0, -2.5):
            for quantity in _FIELDS:
                with self.subTest(
                    scale=scale, quantity=quantity
                ), self.case() as patches:
                    self._case_affine_current_vjps_shrinking_steps(
                        scale, quantity, patches
                    )

    def _case_affine_current_vjps_shrinking_steps(self, scale, quantity, patches):
        """Run one row with case-local objects released before runtime cleanup."""
        curves = _curves()
        for curve in curves:
            curve.fix_all()
        first, total = Current(3.0), Current(10.0)
        total.fix_all()
        coils = [
            Coil(curves[0], first),
            Coil(curves[1], scale * (total - first)),
        ]
        adapter = JaxBiotSavart(coils).set_points(_POINTS)
        cotangent = _cotangent(quantity)
        dofs = np.array(adapter.x, dtype=np.float64)
        patches.callback(setattr, adapter, "x", dofs)
        analytic = _gradient(_vjp(adapter, quantity, cotangent), adapter)
        for exponent in range(5):
            step = 1e-3 * 0.5**exponent
            adapter.x = dofs + step
            plus = np.sum(getattr(adapter, quantity)() * cotangent)
            adapter.x = dofs - step
            minus = np.sum(getattr(adapter, quantity)() * cotangent)
            np.testing.assert_allclose(
                analytic,
                (plus - minus) / (2 * step),
                rtol=1e-9,
                atol=1e-14,
            )

    def test_derivative_retains_fixed_partials(self):
        """Field VJPs retain native partials for fixed owners while exposing the correct
        free gradient."""
        for fixed in ["partial", "curve", "all"]:
            for quantity in _FIELDS:
                with self.subTest(fixed=fixed, quantity=quantity), self.case():
                    self._case_derivative_retains_fixed_partials(fixed, quantity)

    def _case_derivative_retains_fixed_partials(self, fixed, quantity):
        """Run one row with case-local objects released before runtime cleanup."""
        curve = _curves()[0]
        current = Current(1e5)
        curve.fix(0)
        current.fix_all()
        if fixed != "partial":
            curve.fix_all()
        other = _curves()[1]
        if fixed == "all":
            other.fix_all()
        coils = [Coil(curve, current), Coil(other, Current(2e4))]
        if fixed == "all":
            coils[1].current.fix_all()
        native = BiotSavart(coils).set_points(_POINTS)
        adapter = JaxBiotSavart(coils).set_points(_POINTS)
        expected = cast(
            Derivative,
            _vjp(native, quantity, _cotangent(quantity))(native, as_derivative=True),
        )
        derivative = _vjp(adapter, quantity, _cotangent(quantity))
        actual = cast(Derivative, derivative(adapter, as_derivative=True))
        for owner in expected.data:
            np.testing.assert_allclose(
                actual.data[owner],
                expected.data[owner],
                rtol=1e-11,
                atol=1e-13,
            )
        np.testing.assert_allclose(
            _gradient(derivative, adapter),
            _gradient(_vjp(native, quantity, _cotangent(quantity)), native),
            rtol=1e-11,
            atol=1e-13,
        )

    def test_native_symmetry_curve_converts_for_fields_and_vjps(self):
        """Native symmetry Fourier curves yield matching JAX fields and VJPs with or
        without stellarator symmetry."""
        for stellsym in [False, True]:
            with self.subTest(stellsym=stellsym), self.case():
                self._case_native_symmetry_curve_converts_for_fields_and_vjps(stellsym)

    def _case_native_symmetry_curve_converts_for_fields_and_vjps(self, stellsym):
        """Run one row with case-local objects released before runtime cleanup."""
        curve = CurveXYZFourierSymmetries(64, 2, nfp=3, stellsym=stellsym, ntor=2)
        dofs = np.random.default_rng(1729).normal(0.0, 0.03, curve.local_full_dof_size)
        dofs[0] = 1.0
        curve.local_full_x = dofs
        coils = [Coil(curve, Current(1e5))]
        native = BiotSavart(coils).set_points(_POINTS)
        adapter = JaxBiotSavart(coils).set_points(_POINTS)
        for quantity in _FIELDS:
            actual, expected = (
                np.asarray(getattr(adapter, quantity)()),
                getattr(native, quantity)(),
            )
            np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-13)
            np.testing.assert_allclose(
                _gradient(_vjp(adapter, quantity, _cotangent(quantity)), adapter),
                _gradient(_vjp(native, quantity, _cotangent(quantity)), native),
                rtol=1e-11,
                atol=1e-13,
            )

    def test_composed_curves_match_native_fields_owner_vjps_and_finite_difference(self):
        """Nested perturbed and filament curves match native fields, owner partials and
        directional finite differences."""
        for wrapper in _COMPOSED_WRAPPERS:
            for fixed in [False, True]:
                for quantity in _FIELDS:
                    with self.subTest(
                        wrapper=wrapper, fixed=fixed, quantity=quantity
                    ), self.case():
                        self._case_composed_curves_match_native_fields_owner_vjps_and_finite_difference(
                            wrapper, fixed, quantity
                        )

    def _case_composed_curves_match_native_fields_owner_vjps_and_finite_difference(
        self, wrapper, fixed, quantity
    ):
        """Run one row with case-local objects released before runtime cleanup."""
        native, curve, rotation = _composed_native_field(wrapper, fixed)
        expected_field = getattr(native, quantity)().copy()
        cotangent = _cotangent(quantity)
        expected_vjp = _vjp(native, quantity, cotangent)
        expected_gradient = _gradient(expected_vjp, native)
        expected_partials = cast(Derivative, expected_vjp(native, as_derivative=True))
        adapter = JaxBiotSavart(native.coils).set_points(_POINTS)
        np.testing.assert_allclose(
            getattr(adapter, quantity)(),
            expected_field,
            rtol=1e-12,
            atol=1e-14,
        )
        actual_vjp = _vjp(adapter, quantity, cotangent)
        actual_gradient = _gradient(actual_vjp, adapter)
        actual_partials = cast(Derivative, actual_vjp(adapter, as_derivative=True))
        np.testing.assert_allclose(
            actual_gradient,
            expected_gradient,
            rtol=1e-11,
            atol=1e-13,
        )
        self.assertTrue(
            curve in actual_partials.data,
            "curve in actual_partials.data",
        )
        if wrapper != "nested_perturbed":
            self.assertTrue(
                rotation in actual_partials.data,
                "rotation in actual_partials.data",
            )
        for owner in expected_partials.data:
            np.testing.assert_allclose(
                actual_partials.data[owner],
                expected_partials.data[owner],
                rtol=1e-11,
                atol=1e-13,
            )
        pullback = getattr(adapter, quantity + "_pullback_native")(cotangent)
        flat = adapter.coil_cotangents_to_dofs_gradient(
            pullback.d_coil_arrays,
            pullback.coil_indices,
        )
        np.testing.assert_allclose(flat, expected_gradient, rtol=1e-11, atol=1e-13)
        if quantity in ("A", "B"):
            # Both halves of the combined VJP, with both cotangents nonzero, match
            # native; the value half ignores the gradient cotangent.
            combined_vjp = f"{quantity}_and_d{quantity}_vjp"
            vgrad = _cotangent(f"d{quantity}_by_dX")
            getattr(native, f"d{quantity}_by_dX")()
            native_halves = getattr(native, combined_vjp)(cotangent, vgrad)
            adapter_halves = getattr(adapter, combined_vjp)(cotangent, vgrad)
            for half, (native_half, adapter_half) in enumerate(
                zip(native_halves, adapter_halves, strict=True)
            ):
                combined_native = cast(
                    Derivative, native_half(native, as_derivative=True)
                )
                combined_adapter = cast(
                    Derivative, adapter_half(adapter, as_derivative=True)
                )
                for owner in combined_native.data:
                    np.testing.assert_allclose(
                        combined_adapter.data[owner],
                        combined_native.data[owner],
                        rtol=1e-11,
                        atol=1e-13,
                        err_msg=f"{combined_vjp} half {half}",
                    )
                    if half == 0:
                        np.testing.assert_allclose(
                            combined_adapter.data[owner],
                            actual_partials.data[owner],
                            rtol=1e-11,
                            atol=1e-13,
                        )

        dofs = native.x.copy()
        direction = np.linspace(-0.3, 0.4, dofs.size)
        step = 1e-4
        objectives = []
        for multiple in (2, 1, -1, -2):
            native.x = dofs + multiple * step * direction
            perturbed_field = getattr(native, quantity)()
            np.testing.assert_allclose(
                getattr(adapter, quantity)(),
                perturbed_field,
                rtol=1e-12,
                atol=1e-14,
            )
            objectives.append(np.sum(perturbed_field * cotangent))
        native.x = dofs
        finite_difference = (
            -objectives[0] + 8 * objectives[1] - 8 * objectives[2] + objectives[3]
        ) / (12 * step)
        np.testing.assert_allclose(
            actual_gradient @ direction,
            finite_difference,
            rtol=1e-9,
            atol=1e-14,
        )

    def test_raw_kernels_evaluate_single_precision_inputs_in_float64(self):
        """Raw kernels and the B pullback upcast float32 inputs and compute in float64.

        The float32 inputs are exact in float64, so each result must equal, bit
        for bit, the evaluation of their float64 upcast.
        """
        coils = [Coil(curve, Current(1e5)) for curve in _curves()]
        single = (
            _POINTS.astype(np.float32),
            _cotangent("B").astype(np.float32),
            np.stack([coil.curve.gamma() for coil in coils]).astype(np.float32),
            np.stack([coil.curve.gammadash() for coil in coils]).astype(np.float32),
            np.array([coil.current.get_value() for coil in coils], dtype=np.float32),
        )
        double = tuple(value.astype(np.float64) for value in single)

        def evaluate(name, points, cotangent, *coil_data):
            if name == "biot_savart_B_vjp":
                return core_biotsavart.biot_savart_B_vjp(points, cotangent, *coil_data)
            return getattr(core_biotsavart, name)(points, *coil_data)

        for name in (
            "biot_savart_B",
            "biot_savart_A",
            "biot_savart_dB_by_dX",
            "biot_savart_B_and_dB",
            "biot_savart_B_vjp",
        ):
            with self.subTest(kernel=name):
                for actual, expected in zip(
                    jax.tree.leaves(evaluate(name, *single)),
                    jax.tree.leaves(evaluate(name, *double)),
                    strict=True,
                ):
                    self.assertEqual(actual.dtype, np.float64)
                    np.testing.assert_array_equal(
                        np.asarray(actual), np.asarray(expected)
                    )

    def test_grouped_field_evaluates_single_precision_points_in_float64(self):
        """Grouped B and its pullback evaluate float32 points and cotangents in float64."""
        coils = [Coil(curve, Current(1e5)) for curve in _curves()]
        group = (
            jnp.asarray(np.stack([coil.curve.gamma() for coil in coils])),
            jnp.asarray(np.stack([coil.curve.gammadash() for coil in coils])),
            jnp.asarray([coil.current.get_value() for coil in coils]),
        )
        points = _POINTS.astype(np.float32)
        cotangent = _cotangent("B").astype(np.float32)
        native = BiotSavart(coils).set_points(points.astype(np.float64))

        field = cast(
            jax.Array,
            grouped_biot_savart_B_from_spec(
                points, grouped_coil_set_spec_from_inputs([group])
            ),
        )

        self.assertEqual(field.dtype, np.float64, "B must be evaluated in float64")
        np.testing.assert_allclose(
            np.asarray(field),
            native.B(),
            rtol=1e-12,
            atol=1e-14,
            err_msg="float32 points must not lower the coil geometry to float32",
        )
        pullbacks = group_biot_savart_B_vjp(points, cotangent, *group)
        double_pullbacks = group_biot_savart_B_vjp(
            points.astype(np.float64), cotangent.astype(np.float64), *group
        )
        for pullback, double_pullback in zip(pullbacks, double_pullbacks, strict=True):
            self.assertEqual(
                pullback.dtype, np.float64, "pullbacks must be evaluated in float64"
            )
            np.testing.assert_array_equal(
                np.asarray(pullback), np.asarray(double_pullback)
            )


class TestBiotSavartJaxKernelTiling(JaxTestCase):
    """Tiled and fused kernels against the untiled kernels and autodiff, and the
    compiled form of their per-coil reductions.
    """

    def test_fused_B_and_dB_matches_autodiff_of_B(self):
        """Claim: the hand-written fused B + dB/dX integrand, kept as a mirror of
        native ``biot_savart_kernel`` because it is faster, equals the B integrand
        and its ``jax.jacfwd`` in the evaluation point to round-off at every
        quadrature node, and the tiled ``biot_savart_B_and_dB`` equals
        ``biot_savart_B`` and its ``jax.jacfwd`` kernel ``biot_savart_dB_by_dX``.

        Configurations: NCSX coils at points around the magnetic axis, at points
        1-10 mm from a coil, and at points about 10 m away. Integrand errors are
        relative to the node's term size, |gammadash| / r**2 for B and
        |gammadash| / r**3 for dB, so cancellation in the cross product does not
        inflate them; kernel errors are relative to the largest entry at each
        point.
        """
        _bs, axis_points, gammas_np, gds_np, currents_np = (
            _ncsx_biotsavart_parity_fixture()
        )
        gammas = jnp.asarray(gammas_np)
        gammadashs = jnp.asarray(gds_np)
        currents = jnp.asarray(currents_np)
        tangent = gds_np[0, 0] / np.linalg.norm(gds_np[0, 0])
        offset = np.cross(tangent, [0.0, 0.0, 1.0])
        offset /= np.linalg.norm(offset)
        configurations = {
            "around the axis": axis_points,
            "near a coil": gammas_np[0, 0] + np.outer([1e-3, 3e-3, 1e-2], offset),
            "far away": np.array([[10.0, 3.0, -2.0], [-7.0, 8.0, 5.0]]),
        }

        def B_integrand(x):
            return jnp.stack(
                core_biotsavart._biot_savart_B_integrand(x, gammas, gammadashs),
                axis=-1,
            )

        def autodiff_integrand(x):
            # jacfwd yields [..., component, direction]; the fused integrand
            # puts the derivative direction first.
            return B_integrand(x), jnp.swapaxes(jax.jacfwd(B_integrand)(x), -1, -2)

        def fused_integrand(x):
            return core_biotsavart._biot_savart_B_and_dB_integrand(
                x, gammas, gammadashs
            )

        for name, points_np in configurations.items():
            with self.subTest(configuration=name):
                points = jnp.asarray(points_np)
                B_fused, dB_fused = jax.vmap(fused_integrand)(points)
                B_auto, dB_auto = jax.vmap(autodiff_integrand)(points)
                distance = np.linalg.norm(
                    points_np[:, None, None, :] - gammas_np[None], axis=-1
                )
                B_scale = np.linalg.norm(gds_np, axis=-1)[None] / distance**2
                B_error = np.max(np.abs(np.asarray(B_fused - B_auto)), axis=-1)
                dB_error = np.max(np.abs(np.asarray(dB_fused - dB_auto)), axis=(-2, -1))
                self.assertLess(np.max(B_error / B_scale), _ROUND_OFF)
                self.assertLess(np.max(dB_error * distance / B_scale), _ROUND_OFF)

                B_kernel, dB_kernel = core_biotsavart.biot_savart_B_and_dB(
                    points, gammas, gammadashs, currents
                )
                B_reference = np.asarray(
                    biot_savart_B(points, gammas, gammadashs, currents)
                )
                dB_reference = np.asarray(
                    biot_savart_dB_by_dX(points, gammas, gammadashs, currents)
                )
                self.assertLess(
                    np.max(
                        np.max(np.abs(np.asarray(B_kernel) - B_reference), axis=-1)
                        / np.max(np.abs(B_reference), axis=-1)
                    ),
                    _ROUND_OFF,
                )
                self.assertLess(
                    np.max(
                        np.max(np.abs(np.asarray(dB_kernel) - dB_reference), axis=(-2, -1))
                        / np.max(np.abs(dB_reference), axis=(-2, -1))
                    ),
                    _ROUND_OFF,
                )

    def test_forward_kernels_take_their_paths_and_match_untiled(self):
        """B, A, their point Jacobians and Hessians and the fused B and dB/dX
        kernel run the loops each tiling row states, and match the untiled
        kernel, which runs none, for two blocks, three blocks plus a tail and
        coil chunk tails."""
        for integrand, diff_mode in _FORWARD_KERNELS:
            kernel = {"integrand": integrand, "diff_mode": diff_mode}
            # The reference is computed outside a subTest, so no row compares
            # against another kernel's reference if computing it fails.
            with self.case():
                loops, untiled = _loops_and_result(
                    _forward_kernel(integrand, diff_mode, 0, 0), _FORWARD_INPUTS
                )
            with self.subTest(**kernel, tiling="untiled"):
                self.assertEqual(loops, ())
            for name, coil_chunk_size, quadrature_block_size, trip_counts in _TILINGS:
                with self.subTest(**kernel, tiling=name), self.case():
                    loops, tiled = _loops_and_result(
                        _forward_kernel(
                            integrand, diff_mode, coil_chunk_size, quadrature_block_size
                        ),
                        _FORWARD_INPUTS,
                    )
                    self.assertEqual(loops, trip_counts)
                    _assert_results_match(tiled, untiled)

    def test_B_vjp_matches_untiled_across_block_boundaries(self):
        """The B cotangents for positions, tangents and currents match the
        untiled VJP kernel for every tiling row."""
        with self.case():
            untiled = _B_vjp(0, 0)
        for name, coil_chunk_size, quadrature_block_size, _ in _TILINGS:
            with self.subTest(tiling=name), self.case():
                _assert_results_match(
                    _B_vjp(coil_chunk_size, quadrature_block_size), untiled
                )

    def test_hessian_point_chunk_size_is_one_eighth_and_keeps_disabled(self):
        """Hessian tiles use one eighth of positive point tuning, floored at one, while
        zero disables tiling."""
        for tuned, expected in [
            (0, 0),
            (1, 1),
            (7, 1),
            (8, 1),
            (128, 16),
            (2048, 256),
            (4096, 512),
        ]:
            with self.subTest(tuned=tuned, expected=expected), self.case():
                self._case_hessian_point_chunk_size_is_one_eighth_and_keeps_disabled(
                    tuned, expected
                )

    def _case_hessian_point_chunk_size_is_one_eighth_and_keeps_disabled(
        self, tuned, expected
    ):
        """Run one row with case-local objects released before runtime cleanup."""
        self.assertTrue(
            core_biotsavart.hessian_point_chunk_size(tuned) == expected,
            "core_biotsavart.hessian_point_chunk_size(tuned) == expected",
        )

    def test_d2B_kernel_tiled_over_points_matches_untiled(self):
        """Point-tiled Hessians match dense results, and value/Jacobian kernels retain
        their original tile size."""
        rng = np.random.default_rng(7)
        gammas, gammadashs, currents = _coil_set(rng)
        points = jnp.asarray(rng.uniform(-0.4, 0.4, size=(37, 3)))
        untiled = core_biotsavart._make_kernel(
            core_biotsavart._Integrand.B,
            core_biotsavart._DiffMode.HESSIAN,
            0,
            0,
            0,
        )(points, gammas, gammadashs, currents)
        # A tuned size of 64 becomes an 8-point Hessian tile: five tiles of 8 plus a
        # 5-point tail, so both loop bodies and the tail are exercised.
        tiled = core_biotsavart._make_kernel(
            core_biotsavart._Integrand.B,
            core_biotsavart._DiffMode.HESSIAN,
            0,
            0,
            64,
        )(points, gammas, gammadashs, currents)
        self.assertTrue(tiled.shape == (37, 3, 3, 3), "tiled.shape == (37, 3, 3, 3)")
        np.testing.assert_allclose(
            np.asarray(tiled), np.asarray(untiled), rtol=1e-13, atol=1e-16
        )
        b_and_db_tiled = core_biotsavart._make_kernel(
            core_biotsavart._Integrand.B,
            core_biotsavart._DiffMode.VALUE_AND_JACOBIAN,
            0,
            0,
            64,
        )(points, gammas, gammadashs, currents)
        # The non-Hessian kernels keep the tuned size: 37 points fit one 64-point
        # tile, so their result must equal the untiled evaluation exactly.
        b_and_db_untiled = core_biotsavart._make_kernel(
            core_biotsavart._Integrand.B,
            core_biotsavart._DiffMode.VALUE_AND_JACOBIAN,
            0,
            0,
            0,
        )(points, gammas, gammadashs, currents)
        for tiled_leaf, untiled_leaf in zip(
            b_and_db_tiled, b_and_db_untiled, strict=True
        ):
            np.testing.assert_array_equal(
                np.asarray(tiled_leaf), np.asarray(untiled_leaf)
            )

    def test_per_coil_reduces_keep_no_minor_component_axis(self):
        """No per-coil reduce of the compiled B-VJP or dB/dX kernel keeps a
        length-3 component axis minor."""
        for kernel_name in ("B_vjp", "dB_by_dX"):
            for tuning in _TUNINGS:
                with self.subTest(kernel=kernel_name, tuning=tuning), self.case():
                    self._case_per_coil_reduces_keep_no_minor_component_axis(
                        kernel_name, tuning
                    )

    def _case_per_coil_reduces_keep_no_minor_component_axis(self, kernel_name, tuning):
        """Run one row with case-local objects released before runtime cleanup."""
        reduces = _per_coil_reduces(
            _compiled_hlo(kernel_name, tuning, _SIZES), _LAYOUT_COIL_COUNT
        )
        # The guard is vacuous if the parser finds no per-coil reduce at all.
        self.assertTrue(reduces, "no per-coil reduce found in the optimized HLO")
        offending = [
            f"{shape} dimensions={sorted(reduced)}"
            for shape, dims, minor, reduced in reduces.values()
            if dims[minor] == 3 and minor not in reduced
        ]
        self.assertEqual(offending, [], "per-coil reduces with a minor 3-axis")

    def test_per_coil_quadrature_sums_feed_only_their_contraction(self):
        """In the traced B, dB/dX and B-VJP kernels each per-coil quadrature sum
        goes straight into its contraction, at this module's sizes and at the
        fast-lane tiling."""
        cases = (
            (_SIZES, _TUNINGS[0]),
            (_SIZES, _TUNINGS[1]),
            (_FAST_LANE_SIZES, _FAST_LANE_TUNING),
        )
        for kernel_name in ("B", "dB_by_dX", "B_vjp"):
            for sizes, tuning in cases:
                with self.subTest(kernel=kernel_name, sizes=sizes, tuning=tuning):
                    with self.case():
                        self._case_per_coil_quadrature_sums_feed_only_their_contraction(
                            kernel_name, sizes, tuning
                        )

    def _case_per_coil_quadrature_sums_feed_only_their_contraction(
        self, kernel_name, sizes, tuning
    ):
        """Run one row with case-local objects released before runtime cleanup."""
        coils, nodes, _ = sizes
        kernel, arguments = _kernel_and_arguments(kernel_name, tuning, sizes)
        # disable_jit inlines jit calls and unrolls fori_loops, so each sum and
        # every equation that reads it, in the VJP too, are in one jaxpr.
        with jax.disable_jit():
            traced = jax.make_jaxpr(kernel)(*arguments)
        sums = list(_per_coil_quadrature_sums(traced.jaxpr, coils))
        # One sum per component and quadrature block for each traced point
        # chunk and derivative pass; none found would make the guard vacuous.
        block_size = tuning[1]
        blocks = 1 if block_size <= 0 else (nodes + block_size - 1) // block_size
        self.assertTrue(
            sums and len(sums) % (3 * blocks) == 0,
            f"{len(sums)} per-coil quadrature sums for 3 components x {blocks} blocks",
        )
        offending = [
            f"{total.aval.shape} -> "
            + ", ".join(
                "result" if eqn is None else eqn.primitive.name for eqn in reads
            )
            for total, reads in sums
            if not reads or not all(map(_is_contraction, reads))
        ]
        self.assertEqual(
            offending, [], "per-coil quadrature sums used before their contraction"
        )


class TestBiotSavartJaxCoilDofMaps(JaxTestCase):
    """Captured coil-DOF maps keep segmented dependencies and exact input order,
    and pullbacks keep their leaf order and static coil indices.
    """

    def setUp(self):
        super().setUp()
        self.coil_graph = coil_graph()
        self.addCleanup(delattr, self, "coil_graph")

    def test_segments_from_free_positions_merge_runs_and_split_at_fixed_dofs(self):
        """Extraction segments merge contiguous free positions and split correctly at
        fixed DOFs, including empty input."""
        self.assertTrue(
            _owner_segments_from_free_positions(5, [0, 1, 2]) == ((5, 8, 0, 3),),
            "_owner_segments_from_free_positions(5, [0, 1, 2]) == ((5, 8, 0, 3),)",
        )
        self.assertTrue(
            _owner_segments_from_free_positions(2, [0, 1, 2, 4, 5, 6, 7])
            == (
                (2, 5, 0, 3),
                (5, 9, 4, 8),
            ),
            "_owner_segments_from_free_positions(2, [0, 1, 2, 4, 5, 6, 7]) == ( (2, 5, 0, 3), (5, 9, 4, 8), )",
        )
        self.assertTrue(
            _owner_segments_from_free_positions(0, [1, 3, 5])
            == (
                (0, 1, 1, 2),
                (1, 2, 3, 4),
                (2, 3, 5, 6),
            ),
            "_owner_segments_from_free_positions(0, [1, 3, 5]) == ( (0, 1, 1, 2), (1, 2, 3, 4), (2, 3, 5, 6), )",
        )
        self.assertTrue(
            _owner_segments_from_free_positions(7, []) == (),
            "_owner_segments_from_free_positions(7, []) == ()",
        )

    def test_extraction_spec_carries_one_segment_per_run(self):
        """Curve and current extraction maps contain one copy segment per contiguous
        free-DOF run."""
        coil_graph = self.coil_graph
        curves, currents, bs_jax, _oracle = coil_graph
        spec = bs_jax.coil_dof_extraction_spec()

        split_start = _owner_start(bs_jax, curves[0])
        self.assertTrue(
            spec.coils[0].curve_map.owner_segments
            == (
                (split_start, split_start + 3, 0, 3),
                (split_start + 3, split_start + 7, 4, 8),
            ),
            "spec.coils[0].curve_map.owner_segments == ( (split_start, split_start + 3, 0, 3), (split_start + 3, split_start + 7, 4, 8), )",
        )
        intact_start = _owner_start(bs_jax, curves[1])
        self.assertTrue(
            spec.coils[1].curve_map.owner_segments
            == ((intact_start, intact_start + 9, 0, 9),),
            "spec.coils[1].curve_map.owner_segments == ( (intact_start, intact_start + 9, 0, 9), )",
        )
        current_start = _owner_start(bs_jax, currents[0])
        self.assertTrue(
            spec.coils[0].current_map.owner_segments
            == ((current_start, current_start + 1, 0, 1),),
            "spec.coils[0].current_map.owner_segments == ( (current_start, current_start + 1, 0, 1), )",
        )
        self.assertTrue(
            spec.coils[1].current_map.owner_segments == (),
            "spec.coils[1].current_map.owner_segments == ()",
        )

    def test_coil_specs_from_dofs_match_the_optimizable_setter_bitwise(self):
        """Compiled reconstruction matches native DOF setters bitwise under the strict
        transfer guard."""
        coil_graph = self.coil_graph
        curves, currents, bs_jax, oracle = coil_graph
        spec = bs_jax.coil_dof_extraction_spec()
        rng = np.random.default_rng(20260914)
        free_dofs = rng.normal(size=bs_jax.dof_size)

        oracle.x = free_dofs
        expected_curve_dofs = [
            np.asarray(curve.local_full_x).copy() for curve in curves
        ]
        expected_currents = [float(current.get_value()) for current in currents]

        # The guard refuses implicit transfers, so the DOFs are placed before it.
        device_dofs = jnp.asarray(free_dofs)
        compiled = (
            jit(lambda dofs: coil_specs_from_dof_extraction_spec(spec, dofs))
            .lower(device_dofs)
            .compile()
        )
        with jax.transfer_guard("disallow"):
            coil_specs = compiled(device_dofs)

        for coil_spec, expected_dofs, expected_current in zip(
            coil_specs, expected_curve_dofs, expected_currents
        ):
            self.assertTrue(
                np.array_equal(_bits(coil_spec.curve.dofs), _bits(expected_dofs)),
                "np.array_equal(_bits(coil_spec.curve.dofs), _bits(expected_dofs))",
            )
            self.assertTrue(
                np.array_equal(
                    _bits(jnp.atleast_1d(coil_spec.current.value)),
                    _bits(np.asarray([expected_current])),
                ),
                "np.array_equal( _bits(jnp.atleast_1d(coil_spec.current.value)), _bits(np.asarray([expected_current])), )",
            )

    def test_fully_free_extraction_lowers_without_selector_matmuls(self):
        """Fully free DOF reconstruction lowers without selector matrix multiplications."""
        curves = [CurveXYZFourier(_NQUAD, _ORDER) for _ in range(2)]
        currents = [Current(1.0e4), Current(2.0e4)]
        bs_jax = JaxBiotSavart([Coil(c, i) for c, i in zip(curves, currents)])

        text = _lowered_extraction_text(bs_jax)

        self.assertTrue(
            "stablehlo.dot_general" not in text, '"stablehlo.dot_general" not in text'
        )

    def test_partially_fixed_extraction_lowers_one_placement_matmul_per_run(self):
        """Only the split curve's two runs need a placement; free maps are slices."""
        coil_graph = self.coil_graph
        _curves, _currents, bs_jax, _oracle = coil_graph

        text = _lowered_extraction_text(bs_jax)

        self.assertTrue(
            text.count("stablehlo.dot_general") == 2,
            'text.count("stablehlo.dot_general") == 2',
        )

    def test_gradient_through_the_mapping_matches_the_native_selection(self):
        """Autodiff extraction-map cotangents match the native setter-derived
        selection gradient bitwise."""
        coil_graph = self.coil_graph
        curves, _currents, bs_jax, oracle = coil_graph
        spec = bs_jax.coil_dof_extraction_spec()
        rng = np.random.default_rng(7)
        free_dofs = rng.normal(size=bs_jax.dof_size)
        weights = [rng.normal(size=curve.local_full_dof_size) for curve in curves]

        def weighted_curve_dofs(dofs):
            coil_specs = coil_specs_from_dof_extraction_spec(spec, dofs)
            return sum(
                jnp.dot(jnp.asarray(weight), coil_spec.curve.dofs)
                for weight, coil_spec in zip(weights, coil_specs)
            )

        autodiff_gradient = np.asarray(
            jax.grad(weighted_curve_dofs)(jnp.asarray(free_dofs))
        )

        selection = np.zeros((bs_jax.dof_size, sum(w.size for w in weights)))
        oracle.x = np.zeros(bs_jax.dof_size)
        baseline = np.concatenate([np.asarray(curve.local_full_x) for curve in curves])
        for column in range(bs_jax.dof_size):
            probe = np.zeros(bs_jax.dof_size)
            probe[column] = 1.0
            oracle.x = probe
            selection[column] = (
                np.concatenate([np.asarray(curve.local_full_x) for curve in curves])
                - baseline
            )
        expected_gradient = selection @ np.concatenate(weights)

        self.assertTrue(
            np.array_equal(_bits(autodiff_gradient), _bits(expected_gradient)),
            "np.array_equal(_bits(autodiff_gradient), _bits(expected_gradient))",
        )

    def test_native_pullback_preserves_leaf_order_and_static_coil_indices(self):
        """Grouped pullbacks retain leaf order and static coil indices through flattening
        and JIT."""
        arrays = tuple(jnp.asarray([value], dtype=jnp.float64) for value in range(1, 7))
        pullback = BiotSavartFieldPullback(
            d_coil_arrays=(
                (arrays[0], arrays[1], arrays[2]),
                (arrays[3], arrays[4], arrays[5]),
            ),
            coil_indices=((2, 0), (1,)),
        )
        leaves, treedef = jax.tree_util.tree_flatten(pullback)
        self.assertTrue(
            all(
                observed is expected
                for observed, expected in zip(leaves, arrays, strict=True)
            ),
            "all(observed is expected for observed, expected in zip(leaves, arrays, strict=True))",
        )
        restored = jax.tree_util.tree_unflatten(treedef, leaves)
        self.assertTrue(
            restored.coil_indices == ((2, 0), (1,)),
            "restored.coil_indices == ((2, 0), (1,))",
        )
        compiled = jit(lambda value: jax.tree.map(lambda leaf: 2 * leaf, value))(
            restored
        )
        self.assertTrue(
            compiled.coil_indices == pullback.coil_indices,
            "compiled.coil_indices == pullback.coil_indices",
        )
        for actual, expected in zip(
            jax.tree_util.tree_leaves(compiled), arrays, strict=True
        ):
            np.testing.assert_array_equal(actual, 2 * expected)


class TestStageTwoBiotSavartJax(JaxTestCase):
    """Reduced stage-two optimization with native objectives on the JAX field."""

    def test_native_objective_optimization(self):
        """The JAX field's objective gradient matches native, and two L-BFGS-B
        iterations take at least one step that strictly decreases J with a
        finite final gradient."""
        set_backend("jax", device="cpu")
        s = SurfaceRZFourier.from_nphi_ntheta(nphi=9, ntheta=8)
        s.set_rc(0, 0, 1.0)
        s.set_rc(1, 0, 0.3)
        s.set_zs(1, 0, 0.3)
        base_curves = create_equally_spaced_curves(
            2, s.nfp, stellsym=True, R0=1.0, R1=0.5, order=2, numquadpoints=32
        )
        base_currents = [Current(1e5) for _ in base_curves]
        base_currents[0].fix_all()
        coils = coils_via_symmetries(base_curves, base_currents, s.nfp, stellsym=True)
        points = s.gamma().reshape((-1, 3))

        def objective(field):
            field.set_points(points)
            return SquaredFlux(s, field) + 1e-5 * sum(CurveLength(c) for c in base_curves)

        native_JF = objective(BiotSavart(coils))
        JF = objective(JaxBiotSavart(coils))

        def fun(dofs):
            JF.x = dofs
            return JF.J(), JF.dJ()

        initial_value, initial_gradient = fun(JF.x)
        np.testing.assert_allclose(native_JF.J(), initial_value, rtol=1e-12)
        np.testing.assert_allclose(
            initial_gradient,
            native_JF.dJ(),
            rtol=1e-10,
            atol=1e-12 * np.max(np.abs(initial_gradient)),
        )
        res = minimize(
            fun,
            JF.x,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 2, "maxcor": 10},
        )
        self.assertGreaterEqual(res.nit, 1)
        self.assertLess(res.fun, initial_value)
        self.assertTrue(np.all(np.isfinite(res.jac)))
