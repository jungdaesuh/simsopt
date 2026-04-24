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


jax.tree_util.register_dataclass(
    SurfaceXYZTensorFourierSpec,
    data_fields=["dofs", "quadpoints_phi", "quadpoints_theta", "scatter_indices"],
    meta_fields=["nfp", "stellsym", "mpol", "ntor"],
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
    CurveRZFourierSpec,
    CurvePlanarFourierSpec,
    CurveHelicalSpec,
    CurveCWSFourierRZSpec,
    CurvePerturbedSpec,
    CurveFilamentSpec,
]

CurveSpecKind = Literal[
    "xyz_fourier",
    "rz_fourier",
    "planar_fourier",
    "helical",
    "cws_fourier_rz",
    "perturbed",
    "filament",
]


def curve_spec_kind(spec: CurveSpec) -> CurveSpecKind:
    """Return the closed discriminant for a curve spec variant."""
    if isinstance(spec, CurveXYZFourierSpec):
        return "xyz_fourier"
    if isinstance(spec, CurveRZFourierSpec):
        return "rz_fourier"
    if isinstance(spec, CurvePlanarFourierSpec):
        return "planar_fourier"
    if isinstance(spec, CurveHelicalSpec):
        return "helical"
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
) -> SurfaceXYZTensorFourierSpec:
    mpol_int = int(mpol)
    ntor_int = int(ntor)
    stellsym_bool = bool(stellsym)
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
    )
