"""Immutable pytree specs for the pure JAX kernel layer.

These dataclasses are the stable JAX-facing state boundary for geometry,
field, and fixed-surface kernels. The public ``Optimizable`` wrappers still
own mutable compatibility state and flat-DOF orchestration, but compiled JAX
paths should consume these explicit specs rather than live object graphs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Union

import jax
import jax.numpy as jnp
import numpy as np

from ._math_utils import (
    as_jax_float64 as _as_float64_array,
    as_jax_int32 as _as_int32_array,
    as_runtime_float64 as _as_runtime_float64,
)


@dataclass(frozen=True)
class CurveXYZFourierSpec:
    """Immutable payload for pure JAX CurveXYZFourier geometry."""

    dofs: jax.Array
    quadpoints: jax.Array
    order: int


jax.tree_util.register_dataclass(
    CurveXYZFourierSpec,
    data_fields=["dofs", "quadpoints"],
    meta_fields=["order"],
)


@dataclass(frozen=True)
class OrientedCurveXYZFourierSpec:
    """Immutable payload for pure JAX OrientedCurveXYZFourier geometry."""

    dofs: jax.Array
    quadpoints: jax.Array
    order: int


jax.tree_util.register_dataclass(
    OrientedCurveXYZFourierSpec,
    data_fields=["dofs", "quadpoints"],
    meta_fields=["order"],
)


@dataclass(frozen=True)
class CurveRZFourierSpec:
    """Immutable payload for pure JAX CurveRZFourier geometry."""

    dofs: jax.Array
    quadpoints: jax.Array
    order: int
    nfp: int
    stellsym: bool


jax.tree_util.register_dataclass(
    CurveRZFourierSpec,
    data_fields=["dofs", "quadpoints"],
    meta_fields=["order", "nfp", "stellsym"],
)


@dataclass(frozen=True)
class CurvePlanarFourierSpec:
    """Immutable payload for pure JAX CurvePlanarFourier geometry."""

    dofs: jax.Array
    quadpoints: jax.Array
    order: int


jax.tree_util.register_dataclass(
    CurvePlanarFourierSpec,
    data_fields=["dofs", "quadpoints"],
    meta_fields=["order"],
)


@dataclass(frozen=True)
class CurveHelicalSpec:
    """Immutable payload for pure JAX CurveHelical geometry."""

    dofs: jax.Array
    quadpoints: jax.Array
    order: int
    m: int
    ell: int
    R0: float
    r: float


jax.tree_util.register_dataclass(
    CurveHelicalSpec,
    data_fields=["dofs", "quadpoints"],
    meta_fields=["order", "m", "ell", "R0", "r"],
)


@dataclass(frozen=True)
class CurveXYZFourierSymmetriesSpec:
    """Immutable payload for pure JAX CurveXYZFourierSymmetries geometry.

    Mirrors ``simsopt.geo.curvexyzfouriersymmetries.CurveXYZFourierSymmetries``
    constructor parameters needed by ``jaxXYZFourierSymmetriescurve_pure``.
    ``nfp`` and ``ntor`` must be coprime (enforced at host-side construction;
    the spec is the frozen runtime payload).
    """

    dofs: jax.Array
    quadpoints: jax.Array
    order: int
    nfp: int
    stellsym: bool
    ntor: int


jax.tree_util.register_dataclass(
    CurveXYZFourierSymmetriesSpec,
    data_fields=["dofs", "quadpoints"],
    meta_fields=["order", "nfp", "stellsym", "ntor"],
)


@dataclass(frozen=True)
class SurfaceGarabedianSpec:
    """Immutable payload for pure JAX SurfaceGarabedian geometry.

    Mirrors ``simsopt.geo.surfacegarabedian.SurfaceGarabedian`` constructor
    parameters and the flattened Δ_{m,n} DOF buffer. The downstream
    converter ``garabedian_to_rzfourier_spec`` consumes the Δ array plus
    the (mmin, mmax, nmin, nmax) shape meta to produce a
    ``SurfaceRZFourierSpec`` consumable by the existing item-05 / surface
    RZFourier JAX pipeline.

    Stellsym is hard-coded True on the host class (and on this spec) per
    the upstream contract — non-stellsym Garabedian surfaces would
    require imaginary Δ values, which the upstream class explicitly
    rejects.
    """

    dofs: jax.Array
    quadpoints_phi: jax.Array
    quadpoints_theta: jax.Array
    nfp: int
    mmin: int
    mmax: int
    nmin: int
    nmax: int


jax.tree_util.register_dataclass(
    SurfaceGarabedianSpec,
    data_fields=["dofs", "quadpoints_phi", "quadpoints_theta"],
    meta_fields=["nfp", "mmin", "mmax", "nmin", "nmax"],
)


@dataclass(frozen=True)
class SurfaceHennebergSpec:
    """Immutable payload for pure JAX SurfaceHenneberg geometry.

    Mirrors ``simsopt.geo.surfacehenneberg.SurfaceHenneberg`` state used by
    the ``gamma_impl`` / ``gammadash1_impl`` / ``gammadash2_impl`` kernels
    (see ``surfacehenneberg.py:588-740`` for the CPU oracle and the
    Henneberg-Helander-Drevlak paper *J. Plasma Phys.* 87, 905870503 (2021)
    for the parameterisation).

    DOF families
    ------------
    - ``R0nH``  : shape ``(nmax+1,)``, coefficients of ``cos(nfp·n·φ)`` in
      ``R0H(φ)``. Index ``n`` runs ``0..nmax``.
    - ``Z0nH``  : shape ``(nmax+1,)``, coefficients of ``sin(nfp·n·φ)`` in
      ``Z0H(φ)``. Index ``0`` is always zero (no ``sin(0)`` mode); the slot
      is retained so the host-class flat DOF layout matches.
    - ``bn``    : shape ``(nmax+1,)``, coefficients of ``cos(nfp·n·φ)`` in
      ``b(φ)``. Index ``n`` runs ``0..nmax``.
    - ``rhomn`` : shape ``(mmax+1, 2·nmax+1)``, coefficients of
      ``cos(m·θ + nfp·n·φ - α·φ)`` in ``ρ(θ,φ)``. Column index is
      ``n + nmax``. The ``(m=0, n<=0)`` cells are zero by convention (the
      host class never writes them).

    The discrete ``alpha_fac`` ∈ {-1, 0, +1} is the only freedom in the
    helicity selector ``α = 0.5·nfp·alpha_fac``. Stellsym is hard-coded
    True on the host class (and on this spec).
    """

    R0nH: jax.Array
    Z0nH: jax.Array
    bn: jax.Array
    rhomn: jax.Array
    quadpoints_phi: jax.Array
    quadpoints_theta: jax.Array
    nfp: int
    alpha_fac: int
    mmax: int
    nmax: int


jax.tree_util.register_dataclass(
    SurfaceHennebergSpec,
    data_fields=[
        "R0nH",
        "Z0nH",
        "bn",
        "rhomn",
        "quadpoints_phi",
        "quadpoints_theta",
    ],
    meta_fields=["nfp", "alpha_fac", "mmax", "nmax"],
)


@dataclass(frozen=True)
class OptimizableDofMapSpec:
    """Immutable mapping from an owner's full DOF vector into one nested Optimizable."""

    template_full_dofs: jax.Array
    owner_segments: tuple[tuple[int, int, int, int], ...]
    input_mode: str
    input_start: int
    input_end: int


jax.tree_util.register_dataclass(
    OptimizableDofMapSpec,
    data_fields=["template_full_dofs"],
    meta_fields=["owner_segments", "input_mode", "input_start", "input_end"],
)


@dataclass(frozen=True)
class FrameRotationSpec:
    """Immutable payload for pure JAX FrameRotation evaluation."""

    dofs: jax.Array
    quadpoints: jax.Array
    order: int
    scale: float


jax.tree_util.register_dataclass(
    FrameRotationSpec,
    data_fields=["dofs", "quadpoints"],
    meta_fields=["order", "scale"],
)


@dataclass(frozen=True)
class ZeroRotationSpec:
    """Immutable zero-rotation payload."""

    quadpoints: jax.Array


jax.tree_util.register_dataclass(
    ZeroRotationSpec,
    data_fields=["quadpoints"],
    meta_fields=[],
)


@dataclass(frozen=True)
class CurrentValueSpec:
    """Immutable scalar-current payload."""

    value: jax.Array


jax.tree_util.register_dataclass(
    CurrentValueSpec,
    data_fields=["value"],
    meta_fields=[],
)


@dataclass(frozen=True)
class CoilSymmetrySpec:
    """Immutable rotation/scale payload for symmetric coil replicas."""

    rotmat: jax.Array
    scale: float
    has_rotation: bool


jax.tree_util.register_dataclass(
    CoilSymmetrySpec,
    data_fields=["rotmat"],
    meta_fields=["scale", "has_rotation"],
)


@dataclass(frozen=True)
class CoilSpec:
    """Immutable coil payload: curve identity, current, and spatial placement."""

    curve: CurveSpec
    current: CurrentValueSpec
    symmetry: CoilSymmetrySpec


jax.tree_util.register_dataclass(
    CoilSpec,
    data_fields=["curve", "current", "symmetry"],
    meta_fields=[],
)


@dataclass(frozen=True)
class CoilDofExtractionSpec:
    """Immutable owner-DOF -> coil-spec reconstruction payload."""

    curve: CurveSpec
    curve_map: OptimizableDofMapSpec
    current_map: OptimizableDofMapSpec
    symmetry: CoilSymmetrySpec


jax.tree_util.register_dataclass(
    CoilDofExtractionSpec,
    data_fields=["curve", "curve_map", "current_map", "symmetry"],
    meta_fields=[],
)


@dataclass(frozen=True)
class CoilSetDofExtractionSpec:
    """Immutable owner-DOF -> grouped-coil reconstruction payload."""

    coils: tuple[CoilDofExtractionSpec, ...]


jax.tree_util.register_dataclass(
    CoilSetDofExtractionSpec,
    data_fields=["coils"],
    meta_fields=[],
)


@dataclass(frozen=True)
class FieldEvalSpec:
    """Immutable field-evaluation point cloud."""

    points: jax.Array


jax.tree_util.register_dataclass(
    FieldEvalSpec,
    data_fields=["points"],
    meta_fields=[],
)


@dataclass(frozen=True)
class CoilGroupSpec:
    """One rectangular coil batch with a shared quadrature count."""

    gammas: jax.Array
    gammadashs: jax.Array
    currents: jax.Array
    coil_indices: tuple[int, ...]

    def field_inputs(self) -> tuple[jax.Array, jax.Array, jax.Array]:
        return self.gammas, self.gammadashs, self.currents

    def as_grouped_data(self) -> tuple[jax.Array, jax.Array, jax.Array, list[int]]:
        return self.gammas, self.gammadashs, self.currents, list(self.coil_indices)


jax.tree_util.register_dataclass(
    CoilGroupSpec,
    data_fields=["gammas", "gammadashs", "currents"],
    meta_fields=["coil_indices"],
)


@dataclass(frozen=True)
class GroupedCoilSetSpec:
    """Immutable grouped coil geometry/current payload."""

    groups: tuple[CoilGroupSpec, ...]

    def field_inputs(self) -> tuple[tuple[jax.Array, jax.Array, jax.Array], ...]:
        return tuple(group.field_inputs() for group in self.groups)

    def coil_index_lists(self) -> tuple[tuple[int, ...], ...]:
        return tuple(group.coil_indices for group in self.groups)

    def as_grouped_data(
        self,
    ) -> tuple[tuple[jax.Array, jax.Array, jax.Array, list[int]], ...]:
        return tuple(group.as_grouped_data() for group in self.groups)


jax.tree_util.register_dataclass(
    GroupedCoilSetSpec,
    data_fields=["groups"],
    meta_fields=[],
)


@dataclass(frozen=True)
class BiotSavartSpec:
    """Immutable Biot-Savart restart payload with owner DOF reconstruction."""

    coil_dof_extraction: CoilSetDofExtractionSpec
    coil_dofs: jax.Array


jax.tree_util.register_dataclass(
    BiotSavartSpec,
    data_fields=["coil_dof_extraction", "coil_dofs"],
    meta_fields=[],
)


@dataclass(frozen=True)
class FixedSurfaceFluxSpec:
    """Immutable Stage 2 fixed-surface flux contract."""

    points: jax.Array
    normal: jax.Array
    target: jax.Array
    definition: str
    nphi: int
    ntheta: int


jax.tree_util.register_dataclass(
    FixedSurfaceFluxSpec,
    data_fields=["points", "normal", "target"],
    meta_fields=["definition", "nphi", "ntheta"],
)


@dataclass(frozen=True)
class SurfaceRZFourierSpec:
    """Immutable fixed-surface payload for pure JAX SurfaceRZFourier geometry."""

    rc: jax.Array
    zs: jax.Array
    rs: jax.Array
    zc: jax.Array
    quadpoints_phi: jax.Array
    quadpoints_theta: jax.Array
    nfp: int
    stellsym: bool
    mpol: int
    ntor: int


jax.tree_util.register_dataclass(
    SurfaceRZFourierSpec,
    data_fields=[
        "rc",
        "zs",
        "rs",
        "zc",
        "quadpoints_phi",
        "quadpoints_theta",
    ],
    meta_fields=["nfp", "stellsym", "mpol", "ntor"],
)


@dataclass(frozen=True)
class SurfaceXYZFourierSpec:
    """Immutable fixed-surface payload for pure JAX SurfaceXYZFourier geometry."""

    dofs: jax.Array
    quadpoints_phi: jax.Array
    quadpoints_theta: jax.Array
    scatter_indices: jax.Array
    coeff_template: jax.Array
    nfp: int
    stellsym: bool
    mpol: int
    ntor: int


jax.tree_util.register_dataclass(
    SurfaceXYZFourierSpec,
    data_fields=[
        "dofs",
        "quadpoints_phi",
        "quadpoints_theta",
        "scatter_indices",
        "coeff_template",
    ],
    meta_fields=["nfp", "stellsym", "mpol", "ntor"],
)


@dataclass(frozen=True)
class SurfaceXYZTensorFourierSpec:
    """Immutable fixed-surface payload for pure JAX SurfaceXYZTensorFourier geometry."""

    dofs: jax.Array
    quadpoints_phi: jax.Array
    quadpoints_theta: jax.Array
    scatter_indices: jax.Array
    nfp: int
    stellsym: bool
    mpol: int
    ntor: int
    clamped_dims: tuple[bool, bool, bool] = (False, False, False)


jax.tree_util.register_dataclass(
    SurfaceXYZTensorFourierSpec,
    data_fields=["dofs", "quadpoints_phi", "quadpoints_theta", "scatter_indices"],
    meta_fields=["nfp", "stellsym", "mpol", "ntor", "clamped_dims"],
)


SurfaceSpec = Union[
    SurfaceRZFourierSpec,
    SurfaceXYZFourierSpec,
    SurfaceXYZTensorFourierSpec,
]

SurfaceSpecKind = Literal[
    "rz_fourier",
    "xyz_fourier",
    "xyz_tensor_fourier",
]


def surface_spec_kind(spec: SurfaceSpec) -> SurfaceSpecKind:
    """Return the closed discriminant for a surface spec variant."""
    if isinstance(spec, SurfaceRZFourierSpec):
        return "rz_fourier"
    if isinstance(spec, SurfaceXYZFourierSpec):
        return "xyz_fourier"
    if isinstance(spec, SurfaceXYZTensorFourierSpec):
        return "xyz_tensor_fourier"
    raise TypeError(f"Unsupported surface spec type: {type(spec).__name__}")


@dataclass(frozen=True)
class SingleStageSeedSpec:
    """Immutable startup seed payload for the single-stage JAX runner."""

    surface: SurfaceXYZTensorFourierSpec
    coil_set: GroupedCoilSetSpec
    coil_dof_extraction: CoilSetDofExtractionSpec
    coil_dofs: jax.Array
    boozer_iota: jax.Array
    boozer_G: jax.Array
    target_labels: tuple[str, ...]
    hardware_constants: tuple[tuple[str, float], ...]
    self_intersection_mode: str
    schema_version: int
    num_tf_coils: int
    banana_curve_index: int
    tf_current_A: float
    banana_current_A: float


jax.tree_util.register_dataclass(
    SingleStageSeedSpec,
    data_fields=[
        "surface",
        "coil_set",
        "coil_dof_extraction",
        "coil_dofs",
        "boozer_iota",
        "boozer_G",
    ],
    meta_fields=[
        "target_labels",
        "hardware_constants",
        "self_intersection_mode",
        "schema_version",
        "num_tf_coils",
        "banana_curve_index",
        "tf_current_A",
        "banana_current_A",
    ],
)


@dataclass(frozen=True)
class SingleStageRuntimeSpec:
    """Immutable resolved runtime contract for single-stage JAX optimization."""

    seed: SingleStageSeedSpec
    mpol: int
    ntor: int
    nfp: int
    nphi: int
    ntheta: int


jax.tree_util.register_dataclass(
    SingleStageRuntimeSpec,
    data_fields=["seed"],
    meta_fields=["mpol", "ntor", "nfp", "nphi", "ntheta"],
)


@dataclass(frozen=True)
class CurveCWSFourierRZSpec:
    """Immutable curve-on-RZ-surface payload for pure JAX geometry."""

    dofs: jax.Array
    quadpoints: jax.Array
    surface: SurfaceRZFourierSpec
    order: int
    G: float
    H: float

    def surface_dofs(self) -> jax.Array:
        return surface_rz_fourier_dofs_from_spec(self.surface)


jax.tree_util.register_dataclass(
    CurveCWSFourierRZSpec,
    data_fields=["dofs", "quadpoints", "surface"],
    meta_fields=["order", "G", "H"],
)


RotationSpec = Union[FrameRotationSpec, ZeroRotationSpec]


@dataclass(frozen=True)
class CurvePerturbedSpec:
    """Immutable wrapper payload for a perturbed base curve."""

    dofs: jax.Array
    quadpoints: jax.Array
    base_curve: CurveSpec
    base_curve_map: OptimizableDofMapSpec
    sample_gamma: jax.Array
    sample_gammadash: jax.Array
    sample_gammadashdash: jax.Array
    sample_gammadashdashdash: jax.Array


jax.tree_util.register_dataclass(
    CurvePerturbedSpec,
    data_fields=[
        "dofs",
        "quadpoints",
        "base_curve",
        "base_curve_map",
        "sample_gamma",
        "sample_gammadash",
        "sample_gammadashdash",
        "sample_gammadashdashdash",
    ],
    meta_fields=[],
)


@dataclass(frozen=True)
class CurveFilamentSpec:
    """Immutable wrapper payload for a finite-build filament curve."""

    dofs: jax.Array
    quadpoints: jax.Array
    base_curve: CurveSpec
    base_curve_map: OptimizableDofMapSpec
    rotation: RotationSpec
    rotation_map: OptimizableDofMapSpec
    frame_kind: str
    dn: float
    db: float


jax.tree_util.register_dataclass(
    CurveFilamentSpec,
    data_fields=[
        "dofs",
        "quadpoints",
        "base_curve",
        "base_curve_map",
        "rotation",
        "rotation_map",
    ],
    meta_fields=["frame_kind", "dn", "db"],
)


CurveSpec = Union[
    CurveXYZFourierSpec,
    OrientedCurveXYZFourierSpec,
    CurveRZFourierSpec,
    CurvePlanarFourierSpec,
    CurveHelicalSpec,
    CurveXYZFourierSymmetriesSpec,
    CurveCWSFourierRZSpec,
    CurvePerturbedSpec,
    CurveFilamentSpec,
]

CurveSpecKind = Literal[
    "xyz_fourier",
    "oriented_xyz_fourier",
    "rz_fourier",
    "planar_fourier",
    "helical",
    "xyz_fourier_symmetries",
    "cws_fourier_rz",
    "perturbed",
    "filament",
]


def curve_spec_kind(spec: CurveSpec) -> CurveSpecKind:
    """Return the closed discriminant for a curve spec variant."""
    if isinstance(spec, CurveXYZFourierSpec):
        return "xyz_fourier"
    if isinstance(spec, OrientedCurveXYZFourierSpec):
        return "oriented_xyz_fourier"
    if isinstance(spec, CurveRZFourierSpec):
        return "rz_fourier"
    if isinstance(spec, CurvePlanarFourierSpec):
        return "planar_fourier"
    if isinstance(spec, CurveHelicalSpec):
        return "helical"
    if isinstance(spec, CurveXYZFourierSymmetriesSpec):
        return "xyz_fourier_symmetries"
    if isinstance(spec, CurveCWSFourierRZSpec):
        return "cws_fourier_rz"
    if isinstance(spec, CurvePerturbedSpec):
        return "perturbed"
    if isinstance(spec, CurveFilamentSpec):
        return "filament"
    raise TypeError(f"Unsupported curve spec type: {type(spec).__name__}")


def make_coil_group_spec(
    gammas: object,
    gammadashs: object,
    currents: object,
    coil_indices: object,
) -> CoilGroupSpec:
    return CoilGroupSpec(
        gammas=_as_float64_array(gammas),
        gammadashs=_as_float64_array(gammadashs),
        currents=_as_float64_array(currents),
        coil_indices=tuple(int(index) for index in coil_indices),
    )


def make_curve_xyzfourier_spec(
    *,
    dofs: object,
    quadpoints: object,
    order: int,
) -> CurveXYZFourierSpec:
    return CurveXYZFourierSpec(
        dofs=_as_float64_array(dofs),
        quadpoints=_as_float64_array(quadpoints),
        order=int(order),
    )


def make_oriented_curve_xyzfourier_spec(
    *,
    dofs: object,
    quadpoints: object,
    order: int,
) -> OrientedCurveXYZFourierSpec:
    return OrientedCurveXYZFourierSpec(
        dofs=_as_float64_array(dofs),
        quadpoints=_as_float64_array(quadpoints),
        order=int(order),
    )


def make_curve_rzfourier_spec(
    *,
    dofs: object,
    quadpoints: object,
    order: int,
    nfp: int,
    stellsym: bool,
) -> CurveRZFourierSpec:
    return CurveRZFourierSpec(
        dofs=_as_float64_array(dofs),
        quadpoints=_as_float64_array(quadpoints),
        order=int(order),
        nfp=int(nfp),
        stellsym=bool(stellsym),
    )


def make_curve_xyzfouriersymmetries_spec(
    *,
    dofs: object,
    quadpoints: object,
    order: int,
    nfp: int,
    stellsym: bool,
    ntor: int,
) -> CurveXYZFourierSymmetriesSpec:
    from math import gcd

    nfp_int = int(nfp)
    ntor_int = int(ntor)
    if gcd(ntor_int, nfp_int) != 1:
        raise ValueError(
            "CurveXYZFourierSymmetriesSpec requires nfp and ntor coprime; "
            f"got nfp={nfp_int}, ntor={ntor_int}"
        )
    return CurveXYZFourierSymmetriesSpec(
        dofs=_as_float64_array(dofs),
        quadpoints=_as_float64_array(quadpoints),
        order=int(order),
        nfp=nfp_int,
        stellsym=bool(stellsym),
        ntor=ntor_int,
    )


def make_surface_garabedian_spec(
    *,
    dofs: object,
    quadpoints_phi: object,
    quadpoints_theta: object,
    nfp: int,
    mmin: int,
    mmax: int,
    nmin: int,
    nmax: int,
) -> SurfaceGarabedianSpec:
    """Build a ``SurfaceGarabedianSpec`` from the host-class state.

    The Δ_{m,n} buffer is captured as a flat float64 array; the (mmin,
    mmax, nmin, nmax) shape parameters are static meta_fields so they
    seed the JIT cache as compile keys.
    """
    return SurfaceGarabedianSpec(
        dofs=_as_float64_array(dofs),
        quadpoints_phi=_as_float64_array(quadpoints_phi),
        quadpoints_theta=_as_float64_array(quadpoints_theta),
        nfp=int(nfp),
        mmin=int(mmin),
        mmax=int(mmax),
        nmin=int(nmin),
        nmax=int(nmax),
    )


def _garabedian_to_rzfourier_indices(
    *, mmin: int, mmax: int, nmin: int, nmax: int
) -> tuple[
    int,
    int,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Pre-compute the static (m, n) → Δ index mapping at host time.

    Returns ``(mpol, ntor, idx1, idx2, rc_has1, rc_has2, zs_has1,
    zs_has2)``. The rc and zs gathers differ at the (m=0, n=0) slot:
    rc[0, 0] = Δ[1, 0] (single contribution) but zs[0, 0] = 0, so the
    two output channels need separate masks.

    For all other (m, n) inside the loop range, rc and zs share the
    standard Δ1 ± Δ2 pattern, masked by whether each (1±m, ±n) lookup
    lies inside the Δ rectangle.
    """
    mpol = int(max(1, mmax - 1, 1 - mmin))
    ntor = int(max(nmax, -nmin))
    ndim = nmax - nmin + 1
    width = 2 * ntor + 1
    height = mpol + 1

    delta_idx1 = np.zeros((height, width), dtype=np.int64)
    delta_idx2 = np.zeros((height, width), dtype=np.int64)
    rc_has1 = np.zeros((height, width), dtype=np.bool_)
    rc_has2 = np.zeros((height, width), dtype=np.bool_)
    zs_has1 = np.zeros((height, width), dtype=np.bool_)
    zs_has2 = np.zeros((height, width), dtype=np.bool_)

    # CPU oracle (surfacegarabedian.py:171-184):
    #   rc[0, 0] = Δ[1, 0]                              (set explicitly)
    #   for m in range(mpol+1):
    #       n_start = 1 if m == 0 else -ntor
    #       for n in range(n_start, ntor+1):
    #           Δ1 = Δ[1-m, -n] if in range else 0
    #           Δ2 = Δ[1+m, n] if in range else 0
    #           rc[m, n] = Δ1 + Δ2;  zs[m, n] = Δ1 - Δ2
    # Negative-n entries at m=0 are never written and remain zero.
    # The (m=0, n=0) entry is set from Δ[1, 0] alone — *not* doubled —
    # and zs[0, 0] is never touched (stays 0).
    for m_out in range(height):
        n_start = 1 if m_out == 0 else -ntor
        for n_out in range(n_start, ntor + 1):
            n_out_idx = n_out + ntor

            mA, nA = 1 - m_out, -n_out
            mB, nB = 1 + m_out, n_out

            if mmin <= mA <= mmax and nmin <= nA <= nmax:
                row = mA - mmin
                col = nA - nmin
                delta_idx1[m_out, n_out_idx] = row * ndim + col
                rc_has1[m_out, n_out_idx] = True
                zs_has1[m_out, n_out_idx] = True

            if mmin <= mB <= mmax and nmin <= nB <= nmax:
                row = mB - mmin
                col = nB - nmin
                delta_idx2[m_out, n_out_idx] = row * ndim + col
                rc_has2[m_out, n_out_idx] = True
                zs_has2[m_out, n_out_idx] = True

    # Handle (m=0, n=0) explicitly: rc gets Δ[1, 0]; zs stays 0.
    if mmin <= 1 <= mmax and nmin <= 0 <= nmax:
        col = -nmin
        row = 1 - mmin
        delta_idx1[0, ntor] = row * ndim + col
        rc_has1[0, ntor] = True
        # zs_has1[0, ntor] stays False (zs[0, 0] is never set).

    return (
        mpol,
        ntor,
        delta_idx1,
        delta_idx2,
        rc_has1,
        rc_has2,
        zs_has1,
        zs_has2,
    )


def garabedian_to_rzfourier_spec(
    garabedian_spec: SurfaceGarabedianSpec,
) -> SurfaceRZFourierSpec:
    """Convert a ``SurfaceGarabedianSpec`` to an equivalent ``SurfaceRZFourierSpec``.

    Mirrors ``SurfaceGarabedian.to_RZFourier`` (see
    ``src/simsopt/geo/surfacegarabedian.py:161-186``) but stays pure
    functional / JAX-friendly: the (m, n) -> Δ index mapping is static
    metadata, and the per-mode conversion is a gather followed by the
    same add/subtract algebra as the CPU oracle.

    Stellsym=True is the host-class invariant; the resulting RZ spec
    therefore has all-zero ``rs`` and ``zc`` buffers.
    """
    (
        mpol,
        ntor,
        idx1,
        idx2,
        rc_has1,
        rc_has2,
        zs_has1,
        zs_has2,
    ) = _garabedian_to_rzfourier_indices(
        mmin=garabedian_spec.mmin,
        mmax=garabedian_spec.mmax,
        nmin=garabedian_spec.nmin,
        nmax=garabedian_spec.nmax,
    )

    dofs = garabedian_spec.dofs
    idx1_jax = _as_int32_array(idx1)
    idx2_jax = _as_int32_array(idx2)
    delta1 = jnp.take(dofs, idx1_jax)
    delta2 = jnp.take(dofs, idx2_jax)
    zero = delta1 - delta1

    def mask(mask_array: np.ndarray) -> jax.Array:
        return _as_int32_array(mask_array).astype(jnp.bool_)

    rc = jnp.where(mask(rc_has1), delta1, zero) + jnp.where(mask(rc_has2), delta2, zero)
    zs = jnp.where(mask(zs_has1), delta1, zero) - jnp.where(mask(zs_has2), delta2, zero)

    zero_like_rc = rc - rc
    return SurfaceRZFourierSpec(
        rc=rc,
        zs=zs,
        rs=zero_like_rc,
        zc=zero_like_rc,
        quadpoints_phi=garabedian_spec.quadpoints_phi,
        quadpoints_theta=garabedian_spec.quadpoints_theta,
        nfp=garabedian_spec.nfp,
        stellsym=True,
        mpol=int(mpol),
        ntor=int(ntor),
    )


def make_surface_henneberg_spec(
    *,
    R0nH: object,
    Z0nH: object,
    bn: object,
    rhomn: object,
    quadpoints_phi: object,
    quadpoints_theta: object,
    nfp: int,
    alpha_fac: int,
    mmax: int,
    nmax: int,
) -> SurfaceHennebergSpec:
    """Build a ``SurfaceHennebergSpec`` from host-class state.

    Mirrors the four DOF families of
    ``simsopt.geo.surfacehenneberg.SurfaceHenneberg`` plus the discrete
    ``alpha_fac`` ∈ {-1, 0, +1}. Shape parameters are static meta_fields
    that seed the JIT cache as compile keys.

    Raises
    ------
    ValueError
        If ``alpha_fac`` is not in ``{-1, 0, 1}`` or if any of the DOF
        arrays disagrees with the declared ``(mmax, nmax)`` shape.
    """
    nfp_int = int(nfp)
    if nfp_int < 1:
        raise ValueError(f"nfp must be >= 1, got {nfp_int}")

    mmax_int = int(mmax)
    if mmax_int < 1:
        raise ValueError(f"mmax must be >= 1, got {mmax_int}")

    nmax_int = int(nmax)
    if nmax_int < 0:
        raise ValueError(f"nmax must be >= 0, got {nmax_int}")

    alpha_int = int(alpha_fac)
    if alpha_int not in (-1, 0, 1):
        raise ValueError(f"alpha_fac must be one of -1, 0, +1; got {alpha_int}")

    R0nH_jax = _as_float64_array(R0nH)
    Z0nH_jax = _as_float64_array(Z0nH)
    bn_jax = _as_float64_array(bn)
    rhomn_jax = _as_float64_array(rhomn)

    expected_1d = (nmax_int + 1,)
    expected_2d = (mmax_int + 1, 2 * nmax_int + 1)
    if R0nH_jax.shape != expected_1d:
        raise ValueError(
            f"R0nH shape mismatch: expected {expected_1d}, got {R0nH_jax.shape}"
        )
    if Z0nH_jax.shape != expected_1d:
        raise ValueError(
            f"Z0nH shape mismatch: expected {expected_1d}, got {Z0nH_jax.shape}"
        )
    if bn_jax.shape != expected_1d:
        raise ValueError(
            f"bn shape mismatch: expected {expected_1d}, got {bn_jax.shape}"
        )
    if rhomn_jax.shape != expected_2d:
        raise ValueError(
            f"rhomn shape mismatch: expected {expected_2d}, got {rhomn_jax.shape}"
        )

    return SurfaceHennebergSpec(
        R0nH=R0nH_jax,
        Z0nH=Z0nH_jax,
        bn=bn_jax,
        rhomn=rhomn_jax,
        quadpoints_phi=_as_float64_array(quadpoints_phi),
        quadpoints_theta=_as_float64_array(quadpoints_theta),
        nfp=nfp_int,
        alpha_fac=alpha_int,
        mmax=mmax_int,
        nmax=nmax_int,
    )


def make_curve_planarfourier_spec(
    *,
    dofs: object,
    quadpoints: object,
    order: int,
) -> CurvePlanarFourierSpec:
    return CurvePlanarFourierSpec(
        dofs=_as_float64_array(dofs),
        quadpoints=_as_float64_array(quadpoints),
        order=int(order),
    )


def make_curve_helical_spec(
    *,
    dofs: object,
    quadpoints: object,
    order: int,
    m: int,
    ell: int,
    R0: float,
    r: float,
) -> CurveHelicalSpec:
    return CurveHelicalSpec(
        dofs=_as_float64_array(dofs),
        quadpoints=_as_float64_array(quadpoints),
        order=int(order),
        m=int(m),
        ell=int(ell),
        R0=float(R0),
        r=float(r),
    )


def make_optimizable_dof_map_spec(
    *,
    template_full_dofs: object,
    owner_segments: object,
    input_mode: str,
    input_start: int,
    input_end: int,
) -> OptimizableDofMapSpec:
    return OptimizableDofMapSpec(
        template_full_dofs=_as_float64_array(template_full_dofs),
        owner_segments=tuple(
            (
                int(owner_start),
                int(owner_end),
                int(target_start),
                int(target_end),
            )
            for owner_start, owner_end, target_start, target_end in owner_segments
        ),
        input_mode=str(input_mode),
        input_start=int(input_start),
        input_end=int(input_end),
    )


def make_frame_rotation_spec(
    *,
    dofs: object,
    quadpoints: object,
    order: int,
    scale: float,
) -> FrameRotationSpec:
    return FrameRotationSpec(
        dofs=_as_float64_array(dofs),
        quadpoints=_as_float64_array(quadpoints),
        order=int(order),
        scale=float(scale),
    )


def make_zero_rotation_spec(*, quadpoints: object) -> ZeroRotationSpec:
    return ZeroRotationSpec(quadpoints=_as_float64_array(quadpoints))


def make_curve_cwsfourier_rz_spec(
    *,
    dofs: object,
    quadpoints: object,
    surface: SurfaceRZFourierSpec,
    order: int,
    G: float = 0.0,
    H: float = 0.0,
) -> CurveCWSFourierRZSpec:
    return CurveCWSFourierRZSpec(
        dofs=_as_float64_array(dofs),
        quadpoints=_as_float64_array(quadpoints),
        surface=surface,
        order=int(order),
        G=float(G),
        H=float(H),
    )


def make_curve_perturbed_spec(
    *,
    dofs: object,
    quadpoints: object,
    base_curve: CurveSpec,
    base_curve_map: OptimizableDofMapSpec,
    sample_gamma: object,
    sample_gammadash: object,
    sample_gammadashdash: object,
    sample_gammadashdashdash: object,
) -> CurvePerturbedSpec:
    return CurvePerturbedSpec(
        dofs=_as_float64_array(dofs),
        quadpoints=_as_float64_array(quadpoints),
        base_curve=base_curve,
        base_curve_map=base_curve_map,
        sample_gamma=_as_float64_array(sample_gamma),
        sample_gammadash=_as_float64_array(sample_gammadash),
        sample_gammadashdash=_as_float64_array(sample_gammadashdash),
        sample_gammadashdashdash=_as_float64_array(sample_gammadashdashdash),
    )


def make_curve_filament_spec(
    *,
    dofs: object,
    quadpoints: object,
    base_curve: CurveSpec,
    base_curve_map: OptimizableDofMapSpec,
    rotation: RotationSpec,
    rotation_map: OptimizableDofMapSpec,
    frame_kind: str,
    dn: float,
    db: float,
) -> CurveFilamentSpec:
    return CurveFilamentSpec(
        dofs=_as_float64_array(dofs),
        quadpoints=_as_float64_array(quadpoints),
        base_curve=base_curve,
        base_curve_map=base_curve_map,
        rotation=rotation,
        rotation_map=rotation_map,
        frame_kind=str(frame_kind),
        dn=float(dn),
        db=float(db),
    )


def make_current_value_spec(value: object) -> CurrentValueSpec:
    return CurrentValueSpec(value=_as_float64_array([value]))


def _normalize_rotmat(rotmat: object | None) -> tuple[jax.Array, bool]:
    if rotmat is None:
        return jax.device_put(np.eye(3, dtype=np.float64)), False
    return _as_float64_array(rotmat), True


def make_coil_spec(
    *,
    curve: CurveSpec,
    current: CurrentValueSpec,
    rotmat: object | None = None,
    scale: float = 1.0,
) -> CoilSpec:
    return CoilSpec(
        curve=curve,
        current=current,
        symmetry=make_coil_symmetry_spec(rotmat=rotmat, scale=scale),
    )


def make_coil_symmetry_spec(
    *,
    rotmat: object | None = None,
    scale: float = 1.0,
) -> CoilSymmetrySpec:
    rotmat_jax, has_rotation = _normalize_rotmat(rotmat)
    return CoilSymmetrySpec(
        rotmat=rotmat_jax,
        scale=float(scale),
        has_rotation=has_rotation,
    )


def make_coil_dof_extraction_spec(
    *,
    curve: CurveSpec,
    curve_map: OptimizableDofMapSpec,
    current_map: OptimizableDofMapSpec,
    rotmat: object | None = None,
    scale: float = 1.0,
) -> CoilDofExtractionSpec:
    return CoilDofExtractionSpec(
        curve=curve,
        curve_map=curve_map,
        current_map=current_map,
        symmetry=make_coil_symmetry_spec(rotmat=rotmat, scale=scale),
    )


def make_coil_set_dof_extraction_spec(
    coils: object,
) -> CoilSetDofExtractionSpec:
    return CoilSetDofExtractionSpec(coils=tuple(coils))


def apply_coil_symmetry(
    gamma: jax.Array,
    gammadash: jax.Array,
    current: jax.Array,
    symmetry: CoilSymmetrySpec,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Apply rotation/scale transform to curve geometry and current."""
    if symmetry.has_rotation:
        rotmat = _as_runtime_float64(symmetry.rotmat, reference=gamma)
        gamma = gamma @ rotmat
        gammadash = gammadash @ rotmat
    return (
        gamma,
        gammadash,
        current * _as_runtime_float64(symmetry.scale, reference=current),
    )


def make_field_eval_spec(points: object) -> FieldEvalSpec:
    return FieldEvalSpec(points=_as_float64_array(points))


def make_grouped_coil_set_spec(groups: object) -> GroupedCoilSetSpec:
    group_specs = []
    for group in groups:
        if isinstance(group, CoilGroupSpec):
            group_specs.append(group)
            continue
        gammas, gammadashs, currents, coil_indices = group
        group_specs.append(
            make_coil_group_spec(
                gammas,
                gammadashs,
                currents,
                coil_indices,
            )
        )
    return GroupedCoilSetSpec(groups=tuple(group_specs))


def make_biot_savart_spec(
    *,
    coil_dof_extraction: CoilSetDofExtractionSpec,
    coil_dofs: object,
) -> BiotSavartSpec:
    return BiotSavartSpec(
        coil_dof_extraction=coil_dof_extraction,
        coil_dofs=_as_float64_array(coil_dofs),
    )


def make_single_stage_seed_spec(
    *,
    surface: SurfaceXYZTensorFourierSpec,
    coil_set: GroupedCoilSetSpec,
    coil_dof_extraction: CoilSetDofExtractionSpec,
    coil_dofs: object,
    boozer_iota: object,
    boozer_G: object,
    target_labels: object,
    hardware_constants: object,
    self_intersection_mode: str,
    schema_version: int,
    num_tf_coils: int,
    banana_curve_index: int,
    tf_current_A: float,
    banana_current_A: float,
) -> SingleStageSeedSpec:
    return SingleStageSeedSpec(
        surface=surface,
        coil_set=coil_set,
        coil_dof_extraction=coil_dof_extraction,
        coil_dofs=_as_float64_array(coil_dofs),
        boozer_iota=_as_float64_array([boozer_iota]),
        boozer_G=_as_float64_array([boozer_G]),
        target_labels=tuple(str(label) for label in target_labels),
        hardware_constants=tuple(
            (str(name), float(value)) for name, value in hardware_constants
        ),
        self_intersection_mode=str(self_intersection_mode),
        schema_version=int(schema_version),
        num_tf_coils=int(num_tf_coils),
        banana_curve_index=int(banana_curve_index),
        tf_current_A=float(tf_current_A),
        banana_current_A=float(banana_current_A),
    )


def make_single_stage_runtime_spec(
    *,
    seed: SingleStageSeedSpec,
    mpol: int,
    ntor: int,
    nfp: int,
    nphi: int,
    ntheta: int,
) -> SingleStageRuntimeSpec:
    return SingleStageRuntimeSpec(
        seed=seed,
        mpol=int(mpol),
        ntor=int(ntor),
        nfp=int(nfp),
        nphi=int(nphi),
        ntheta=int(ntheta),
    )


def make_fixed_surface_flux_spec(
    *,
    points: object,
    normal: object,
    target: object,
    definition: str,
) -> FixedSurfaceFluxSpec:
    normal_jax = _as_float64_array(normal)
    return FixedSurfaceFluxSpec(
        points=_as_float64_array(points),
        normal=normal_jax,
        target=_as_float64_array(target),
        definition=definition,
        nphi=int(normal_jax.shape[0]),
        ntheta=int(normal_jax.shape[1]),
    )


def make_surface_rzfourier_spec(
    *,
    rc: object,
    zs: object,
    quadpoints_phi: object,
    quadpoints_theta: object,
    nfp: int,
    stellsym: bool,
    rs: object | None = None,
    zc: object | None = None,
) -> SurfaceRZFourierSpec:
    rc_jax = _as_float64_array(rc)
    zs_jax = _as_float64_array(zs)
    zero_like_rc = rc_jax - rc_jax
    rs_jax = zero_like_rc if rs is None else _as_float64_array(rs)
    zc_jax = zero_like_rc if zc is None else _as_float64_array(zc)
    return SurfaceRZFourierSpec(
        rc=rc_jax,
        zs=zs_jax,
        rs=rs_jax,
        zc=zc_jax,
        quadpoints_phi=_as_float64_array(quadpoints_phi),
        quadpoints_theta=_as_float64_array(quadpoints_theta),
        nfp=int(nfp),
        stellsym=bool(stellsym),
        mpol=int(rc_jax.shape[0] - 1),
        ntor=int((rc_jax.shape[1] - 1) // 2),
    )


def _surface_rz_fourier_block_mode_positions(
    *,
    mpol: int,
    ntor: int,
    include_zero_mode: bool,
) -> np.ndarray:
    width = 2 * ntor + 1
    positions: list[int] = []

    start_n = 0 if include_zero_mode else 1
    for n in range(start_n, ntor + 1):
        positions.append(n + ntor)

    for m in range(1, mpol + 1):
        for n in range(-ntor, ntor + 1):
            positions.append(m * width + n + ntor)

    return np.asarray(positions, dtype=np.int32)


def _surface_rz_fourier_gather_modes(
    coeffs: jax.Array,
    positions: np.ndarray,
    flat_size: int,
) -> jax.Array:
    coeff_vector = jnp.reshape(_as_float64_array(coeffs), (flat_size,))
    return jnp.take(coeff_vector, _as_int32_array(positions), axis=0)


def surface_rz_fourier_dofs_from_spec(spec: SurfaceRZFourierSpec) -> jax.Array:
    include_positions = _surface_rz_fourier_block_mode_positions(
        mpol=spec.mpol,
        ntor=spec.ntor,
        include_zero_mode=True,
    )
    exclude_positions = _surface_rz_fourier_block_mode_positions(
        mpol=spec.mpol,
        ntor=spec.ntor,
        include_zero_mode=False,
    )
    flat_size = int((spec.mpol + 1) * (2 * spec.ntor + 1))
    rc = _surface_rz_fourier_gather_modes(spec.rc, include_positions, flat_size)
    zs = _surface_rz_fourier_gather_modes(spec.zs, exclude_positions, flat_size)
    if spec.stellsym:
        return jnp.concatenate((rc, zs))
    rs = _surface_rz_fourier_gather_modes(spec.rs, exclude_positions, flat_size)
    zc = _surface_rz_fourier_gather_modes(spec.zc, include_positions, flat_size)
    return jnp.concatenate((rc, rs, zc, zs))


def make_surface_xyz_fourier_spec(
    *,
    dofs: object,
    quadpoints_phi: object,
    quadpoints_theta: object,
    nfp: int,
    stellsym: bool,
    mpol: int,
    ntor: int,
) -> SurfaceXYZFourierSpec:
    mpol_int = int(mpol)
    ntor_int = int(ntor)
    stellsym_bool = bool(stellsym)
    return SurfaceXYZFourierSpec(
        dofs=_as_float64_array(dofs),
        quadpoints_phi=_as_float64_array(quadpoints_phi),
        quadpoints_theta=_as_float64_array(quadpoints_theta),
        scatter_indices=_as_int32_array(
            _surface_xyz_fourier_scatter_indices(
                mpol=mpol_int,
                ntor=ntor_int,
                stellsym=stellsym_bool,
            )
        ),
        coeff_template=_as_float64_array(
            np.zeros(6 * (mpol_int + 1) * (2 * ntor_int + 1), dtype=np.float64)
        ),
        nfp=int(nfp),
        stellsym=stellsym_bool,
        mpol=mpol_int,
        ntor=ntor_int,
    )


def _surface_xyz_fourier_scatter_indices(
    *,
    mpol: int,
    ntor: int,
    stellsym: bool,
) -> np.ndarray:
    n_per = int((mpol + 1) * (2 * ntor + 1))
    cos_positions = np.arange(ntor, n_per, dtype=np.int32)
    sin_positions = np.arange(ntor + 1, n_per, dtype=np.int32)

    if stellsym:
        return np.concatenate(
            (
                cos_positions,
                3 * n_per + sin_positions,
                5 * n_per + sin_positions,
            )
        ).astype(np.int32)

    return np.concatenate(
        (
            cos_positions,
            n_per + sin_positions,
            2 * n_per + cos_positions,
            3 * n_per + sin_positions,
            4 * n_per + cos_positions,
            5 * n_per + sin_positions,
        )
    ).astype(np.int32)


def _surface_xyz_tensor_scatter_indices(
    *,
    mpol: int,
    ntor: int,
    stellsym: bool,
    scatter_indices: object | None,
) -> jax.Array:
    if scatter_indices is not None:
        return _as_int32_array(scatter_indices)
    if not stellsym:
        return _as_int32_array(np.zeros((0,), dtype=np.int32))

    from ..geo.surface_fourier_jax import stellsym_scatter_indices

    return _as_int32_array(stellsym_scatter_indices(mpol, ntor))


def make_surface_xyz_tensor_fourier_spec(
    *,
    dofs: object,
    quadpoints_phi: object,
    quadpoints_theta: object,
    nfp: int,
    stellsym: bool,
    mpol: int,
    ntor: int,
    scatter_indices: object | None = None,
    clamped_dims: tuple[bool, bool, bool] = (False, False, False),
) -> SurfaceXYZTensorFourierSpec:
    mpol_int = int(mpol)
    ntor_int = int(ntor)
    stellsym_bool = bool(stellsym)
    clamped_tuple = tuple(bool(flag) for flag in clamped_dims)
    if len(clamped_tuple) != 3:
        raise ValueError(
            "clamped_dims must have exactly 3 boolean flags (x, y, z); "
            f"got length {len(clamped_tuple)}"
        )
    return SurfaceXYZTensorFourierSpec(
        dofs=_as_float64_array(dofs),
        quadpoints_phi=_as_float64_array(quadpoints_phi),
        quadpoints_theta=_as_float64_array(quadpoints_theta),
        scatter_indices=_surface_xyz_tensor_scatter_indices(
            mpol=mpol_int,
            ntor=ntor_int,
            stellsym=stellsym_bool,
            scatter_indices=scatter_indices,
        ),
        nfp=int(nfp),
        stellsym=stellsym_bool,
        mpol=mpol_int,
        ntor=ntor_int,
        clamped_dims=clamped_tuple,
    )
