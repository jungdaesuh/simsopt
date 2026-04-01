"""Immutable specs for the pure JAX kernel layer."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


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
class CurrentValueSpec:
    """Immutable scalar-current payload."""

    value: jax.Array


jax.tree_util.register_dataclass(
    CurrentValueSpec,
    data_fields=["value"],
    meta_fields=[],
)


@dataclass(frozen=True)
class CoilSpec:
    """Immutable coil payload composed from curve/current specs and wrappers."""

    curve: CurveSpec
    current: CurrentValueSpec
    rotmat: jax.Array
    scale: float
    has_rotation: bool


jax.tree_util.register_dataclass(
    CoilSpec,
    data_fields=["curve", "current", "rotmat"],
    meta_fields=["scale", "has_rotation"],
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
class CurveCWSFourierRZSpec:
    """Immutable curve-on-RZ-surface payload for pure JAX geometry."""

    dofs: jax.Array
    quadpoints: jax.Array
    surface: SurfaceRZFourierSpec
    order: int
    G: float
    H: float

    def surface_dofs(self) -> jax.Array:
        from .surface_rzfourier import surface_rz_fourier_dofs_from_spec

        return surface_rz_fourier_dofs_from_spec(self.surface)


jax.tree_util.register_dataclass(
    CurveCWSFourierRZSpec,
    data_fields=["dofs", "quadpoints", "surface"],
    meta_fields=["order", "G", "H"],
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


CurveSpec = CurveXYZFourierSpec | CurveRZFourierSpec | CurveCWSFourierRZSpec


def make_coil_group_spec(
    gammas: object,
    gammadashs: object,
    currents: object,
    coil_indices: object,
) -> CoilGroupSpec:
    return CoilGroupSpec(
        gammas=jnp.asarray(gammas, dtype=jnp.float64),
        gammadashs=jnp.asarray(gammadashs, dtype=jnp.float64),
        currents=jnp.asarray(currents, dtype=jnp.float64),
        coil_indices=tuple(int(index) for index in coil_indices),
    )


def make_curve_xyzfourier_spec(
    *,
    dofs: object,
    quadpoints: object,
    order: int,
) -> CurveXYZFourierSpec:
    return CurveXYZFourierSpec(
        dofs=jnp.asarray(dofs, dtype=jnp.float64),
        quadpoints=jnp.asarray(quadpoints, dtype=jnp.float64),
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
        dofs=jnp.asarray(dofs, dtype=jnp.float64),
        quadpoints=jnp.asarray(quadpoints, dtype=jnp.float64),
        order=int(order),
        nfp=int(nfp),
        stellsym=bool(stellsym),
    )


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
        dofs=jnp.asarray(dofs, dtype=jnp.float64),
        quadpoints=jnp.asarray(quadpoints, dtype=jnp.float64),
        surface=surface,
        order=int(order),
        G=float(G),
        H=float(H),
    )


def make_current_value_spec(value: object) -> CurrentValueSpec:
    return CurrentValueSpec(value=jnp.asarray([value], dtype=jnp.float64))


def _normalize_rotmat(rotmat: object | None) -> tuple[jax.Array, bool]:
    if rotmat is None:
        return jnp.eye(3, dtype=jnp.float64), False
    return jnp.asarray(rotmat, dtype=jnp.float64), True


def make_coil_spec(
    *,
    curve: CurveSpec,
    current: CurrentValueSpec,
    rotmat: object | None = None,
    scale: float = 1.0,
) -> CoilSpec:
    rotmat_jax, has_rotation = _normalize_rotmat(rotmat)
    return CoilSpec(
        curve=curve,
        current=current,
        rotmat=rotmat_jax,
        scale=float(scale),
        has_rotation=has_rotation,
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


def make_field_eval_spec(points: object) -> FieldEvalSpec:
    return FieldEvalSpec(points=jnp.asarray(points, dtype=jnp.float64))


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
    normal_jax = jnp.asarray(normal, dtype=jnp.float64)
    return FixedSurfaceFluxSpec(
        points=jnp.asarray(points, dtype=jnp.float64),
        normal=normal_jax,
        target=jnp.asarray(target, dtype=jnp.float64),
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
    rc_jax = jnp.asarray(rc, dtype=jnp.float64)
    zs_jax = jnp.asarray(zs, dtype=jnp.float64)
    zero_like_rc = jnp.zeros_like(rc_jax)
    rs_jax = zero_like_rc if rs is None else jnp.asarray(rs, dtype=jnp.float64)
    zc_jax = zero_like_rc if zc is None else jnp.asarray(zc, dtype=jnp.float64)
    return SurfaceRZFourierSpec(
        rc=rc_jax,
        zs=zs_jax,
        rs=rs_jax,
        zc=zc_jax,
        quadpoints_phi=jnp.asarray(quadpoints_phi, dtype=jnp.float64),
        quadpoints_theta=jnp.asarray(quadpoints_theta, dtype=jnp.float64),
        nfp=int(nfp),
        stellsym=bool(stellsym),
        mpol=int(rc_jax.shape[0] - 1),
        ntor=int((rc_jax.shape[1] - 1) // 2),
    )
