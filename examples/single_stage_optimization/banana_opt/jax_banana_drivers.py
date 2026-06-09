from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from simsopt._core.derivative import derivative_dec
from simsopt._core.optimizable import Optimizable, load
from simsopt.field import BiotSavart, Coil, Current, coils_via_symmetries
from simsopt.field.coil import ScaledCurrent
from simsopt.geo import (
    BoozerSurface,
    CurveXYZFourier,
    Surface,
    SurfaceRZFourier,
    SurfaceXYZFourier,
    SurfaceXYZTensorFourier,
    Volume,
    create_equally_spaced_curves,
)
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.geo.boozer_surface import BoozerSurfaceJAX
from simsopt_jax_adapters.geo.curvecwsfourier import CurveCWSFourierCPP
from simsopt_jax_adapters.geo.curve_objectives import (
    CurveCurveDistanceJAX,
    CurveLengthJAX,
    CurveSurfaceDistanceJAX,
    LpCurveCurvatureJAX,
)
from simsopt_jax_adapters.geo.surface_objectives import (
    BoozerResidualJAX,
    IotasJAX,
    NonQuasiSymmetricRatioJAX,
)
from simsopt_jax_adapters.objectives.flux import SquaredFluxJAX
from simsopt.objectives import QuadraticPenalty

from .jax_banana_types import (
    BANANA_IDX,
    BANANA_TARGETS,
    DEFAULT_BANANA_DOFS,
    HBT_BANANA_WS,
    HBT_TF,
    HBT_VF,
    HARDWARE_LIMITS,
    N_BANANA,
    BananaDriverPaths,
    BoozerSolveState,
    EvaluationTracker,
    SingleStageWeights,
    Stage2Weights,
    TF_IDX,
    WeightedTerm,
)


MU0 = 4.0e-7 * np.pi
KA_TO_A = 1.0e3


def _as_float(value: object) -> float:
    return float(np.asarray(jax.device_get(value), dtype=np.float64))


def _as_jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "devices"):
        return np.asarray(jax.device_get(value)).tolist()
    return value


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_as_jsonable) + "\n",
        encoding="utf-8",
    )


class DriverLog:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def __call__(self, message: object) -> None:
        text = str(message)
        print(text, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")


def output_paths(output_dir: str | Path, *, prefix: str) -> BananaDriverPaths:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    return BananaDriverPaths(
        biotsavart=root / f"{prefix}_biotsavart.json",
        boozersurface=root / f"{prefix}_boozersurface.json",
        surface=root / f"{prefix}_surface.json",
        results=root / "results.json",
        log=root / f"{prefix}.log",
    )


def ensure_writable_outputs(paths: BananaDriverPaths, *, overwrite: bool) -> None:
    if overwrite:
        return
    existing = [
        path
        for path in (
            paths.biotsavart,
            paths.boozersurface,
            paths.surface,
            paths.results,
            paths.log,
        )
        if path is not None and path.exists()
    ]
    if existing:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing output files: {joined}")


def read_banana_dofs(path: str | Path | None) -> dict[str, float]:
    if path is None:
        return dict(DEFAULT_BANANA_DOFS)
    rows = np.loadtxt(path, dtype=str, ndmin=2, comments="#")
    return {str(key): float(value) for key, value in rows}


def build_surface(
    surface_path: str | Path,
    *,
    vmec_s: float,
    scale: float | None,
    nphi: int,
    ntheta: int,
) -> SurfaceRZFourier | SurfaceXYZFourier | SurfaceXYZTensorFourier:
    path = Path(surface_path)
    suffix = path.suffix
    if suffix == ".json":
        surface = load(str(path))
    elif suffix == ".nc":
        surface = SurfaceRZFourier.from_wout(str(path), s=vmec_s)
        if scale is not None:
            surface.set_dofs(surface.get_dofs() * float(scale) / surface.major_radius())
    else:
        raise ValueError(f"Unsupported surface extension {suffix!r}; expected .json or .nc")
    return resize_surface(surface, nphi=nphi, ntheta=ntheta)


def resize_surface(
    init_surface: SurfaceRZFourier | SurfaceXYZFourier | SurfaceXYZTensorFourier,
    *,
    mpol: int | None = None,
    ntor: int | None = None,
    nphi: int,
    ntheta: int,
) -> SurfaceRZFourier | SurfaceXYZFourier | SurfaceXYZTensorFourier:
    target_mpol = init_surface.mpol if mpol is None else int(mpol)
    target_ntor = init_surface.ntor if ntor is None else int(ntor)
    if isinstance(init_surface, SurfaceRZFourier):
        return init_surface.copy(
            mpol=target_mpol,
            ntor=target_ntor,
            nphi=nphi,
            ntheta=ntheta,
        )

    quadpoints_phi, quadpoints_theta = Surface.get_quadpoints(
        nphi=nphi,
        ntheta=ntheta,
        nfp=init_surface.nfp,
    )
    if isinstance(init_surface, SurfaceXYZFourier):
        source = SurfaceXYZFourier(
            mpol=init_surface.mpol,
            ntor=init_surface.ntor,
            nfp=init_surface.nfp,
            stellsym=init_surface.stellsym,
            quadpoints_phi=quadpoints_phi,
            quadpoints_theta=quadpoints_theta,
        )
        target = SurfaceXYZFourier(
            mpol=target_mpol,
            ntor=target_ntor,
            nfp=init_surface.nfp,
            stellsym=init_surface.stellsym,
            quadpoints_phi=quadpoints_phi,
            quadpoints_theta=quadpoints_theta,
        )
    else:
        source = SurfaceXYZTensorFourier(
            mpol=init_surface.mpol,
            ntor=init_surface.ntor,
            nfp=init_surface.nfp,
            stellsym=init_surface.stellsym,
            clamped_dims=list(init_surface.clamped_dims),
            quadpoints_phi=quadpoints_phi,
            quadpoints_theta=quadpoints_theta,
        )
        target = SurfaceXYZTensorFourier(
            mpol=target_mpol,
            ntor=target_ntor,
            nfp=init_surface.nfp,
            stellsym=init_surface.stellsym,
            clamped_dims=list(init_surface.clamped_dims),
            quadpoints_phi=quadpoints_phi,
            quadpoints_theta=quadpoints_theta,
        )
    source.set_dofs(init_surface.get_dofs())
    target.least_squares_fit(source.gamma())
    return target


def build_boozer_surface_copy(
    init_surface: SurfaceRZFourier | SurfaceXYZFourier | SurfaceXYZTensorFourier,
    *,
    mpol: int,
    ntor: int,
    nphi: int,
    ntheta: int,
) -> SurfaceXYZTensorFourier:
    source = resize_surface(
        init_surface,
        mpol=mpol,
        ntor=ntor,
        nphi=nphi,
        ntheta=ntheta,
    )
    surface = SurfaceXYZTensorFourier(
        mpol=source.mpol,
        ntor=source.ntor,
        nfp=source.nfp,
        stellsym=source.stellsym,
        quadpoints_phi=source.quadpoints_phi,
        quadpoints_theta=source.quadpoints_theta,
    )
    surface.least_squares_fit(source.gamma())
    return surface


def soft_constraint_weight(weight: float) -> float | None:
    return None if weight == 0.0 else weight


def _scaled_current(current_ka: float, *, fix_current: bool) -> ScaledCurrent:
    current = ScaledCurrent(Current(1.0), current_ka * KA_TO_A)
    if fix_current:
        current.fix_all()
    return current


def _fixed_circular_curve(radius: float, z_position: float) -> CurveXYZFourier:
    curve = CurveXYZFourier(128, order=1)
    curve.set("xc(1)", radius)
    curve.set("ys(1)", radius)
    curve.set("zc(0)", z_position)
    curve.fix_all()
    return curve


def generate_tf_coils(tf_current_ka: float, *, fix_current: bool) -> list[Coil]:
    curves = create_equally_spaced_curves(
        HBT_TF.n_coils,
        HBT_TF.nfp,
        HBT_TF.stellsym,
        R0=HBT_TF.major_radius,
        R1=HBT_TF.minor_radius,
        order=HBT_TF.order,
    )
    for curve in curves:
        curve.fix_all()
    current = _scaled_current(tf_current_ka, fix_current=fix_current)
    return [Coil(curve, current) for curve in curves]


def generate_banana_coils(
    *,
    banana_current_ka: float,
    banana_order: int,
    banana_dofs: dict[str, float],
    fix_current: bool,
) -> list[Coil]:
    winding_surface = SurfaceRZFourier(
        nfp=HBT_BANANA_WS.nfp,
        stellsym=HBT_BANANA_WS.stellsym,
    )
    winding_surface.set_rc(0, 0, HBT_BANANA_WS.major_radius)
    winding_surface.set_rc(1, 0, HBT_BANANA_WS.minor_radius)
    winding_surface.set_zs(1, 0, HBT_BANANA_WS.minor_radius)
    quadpoints = np.linspace(0.0, 1.0, 64 * int(banana_order), endpoint=False)
    banana_curve = CurveCWSFourierCPP(
        quadpoints,
        order=int(banana_order),
        surf=winding_surface,
    )
    for key, value in banana_dofs.items():
        banana_curve.set(key, value)
    banana_current = _scaled_current(banana_current_ka, fix_current=fix_current)
    return coils_via_symmetries(
        [banana_curve],
        [banana_current],
        HBT_BANANA_WS.nfp,
        HBT_BANANA_WS.stellsym,
    )


def generate_proxy_coils(proxy_current_ka: float, proxy_rz: tuple[float, float]) -> list[Coil]:
    radius, z_position = proxy_rz
    curve = _fixed_circular_curve(radius, z_position)
    current = _scaled_current(proxy_current_ka, fix_current=True)
    return [Coil(curve, current)]


def generate_vf_coils(vf_current_ka: float, *, fix_current: bool) -> list[Coil]:
    coils: list[Coil] = []
    for radius, z_position, current_sign in zip(
        HBT_VF.rs,
        HBT_VF.zs,
        HBT_VF.sign_curr,
        strict=True,
    ):
        curve = _fixed_circular_curve(radius, z_position)
        current = _scaled_current(
            current_sign * vf_current_ka,
            fix_current=fix_current,
        )
        coils.append(Coil(curve, current))
    return coils


def build_biotsavart(
    *,
    tf_current_ka: float,
    tf_fix_current: bool,
    banana_current_ka: float,
    banana_fix_current: bool,
    banana_order: int,
    banana_dofs: dict[str, float],
    proxy_current_ka: float,
    proxy_rz: tuple[float, float],
    vf_current_ka: float,
    vf_fix_current: bool,
) -> BiotSavart:
    coils: list[Coil] = []
    coils.extend(generate_tf_coils(tf_current_ka, fix_current=tf_fix_current))
    coils.extend(
        generate_banana_coils(
            banana_current_ka=banana_current_ka,
            banana_order=banana_order,
            banana_dofs=banana_dofs,
            fix_current=banana_fix_current,
        )
    )
    coils.extend(generate_proxy_coils(proxy_current_ka, proxy_rz))
    coils.extend(generate_vf_coils(vf_current_ka, fix_current=vf_fix_current))
    return BiotSavart(coils)


def load_or_build_biotsavart(
    *,
    biotsavart_file: str | Path | None,
    tf_current_ka: float,
    tf_fix_current: bool,
    banana_current_ka: float,
    banana_fix_current: bool,
    banana_order: int,
    banana_dofs: dict[str, float],
    proxy_current_ka: float,
    proxy_rz: tuple[float, float],
    vf_current_ka: float,
    vf_fix_current: bool,
) -> BiotSavart:
    if biotsavart_file is not None:
        loaded = load(str(biotsavart_file))
        if not isinstance(loaded, BiotSavart):
            raise TypeError(f"Expected BiotSavart JSON, got {type(loaded)!r}")
        return loaded
    return build_biotsavart(
        tf_current_ka=tf_current_ka,
        tf_fix_current=tf_fix_current,
        banana_current_ka=banana_current_ka,
        banana_fix_current=banana_fix_current,
        banana_order=banana_order,
        banana_dofs=banana_dofs,
        proxy_current_ka=proxy_current_ka,
        proxy_rz=proxy_rz,
        vf_current_ka=vf_current_ka,
        vf_fix_current=vf_fix_current,
    )


def banana_curves(biotsavart: BiotSavart | BiotSavartJAX) -> list[object]:
    return [coil.curve for coil in biotsavart.coils[BANANA_IDX : BANANA_IDX + N_BANANA]]


def banana_base_curve(biotsavart: BiotSavart | BiotSavartJAX) -> object:
    return biotsavart.coils[BANANA_IDX].curve


def tf_current_total_abs_A(biotsavart: BiotSavart | BiotSavartJAX) -> float:
    return float(
        sum(abs(coil.current.get_value()) for coil in biotsavart.coils[TF_IDX:BANANA_IDX])
    )


def current_abs_upper_bound_penalty(current, threshold: float) -> QuadraticPenalty:
    return QuadraticPenalty(CurrentMagnitude(current), threshold, "max")


class CurrentMagnitude(Optimizable):
    def __init__(self, current):
        self.current = current
        super().__init__(depends_on=[current])

    def J(self) -> float:
        return abs(float(self.current.get_value()))

    @derivative_dec
    def dJ(self):
        value = float(self.current.get_value())
        sign = 1.0 if value >= 0.0 else -1.0
        return self.current.vjp(np.asarray([sign], dtype=float))

    return_fn_map = {"J": J, "dJ": dJ}


@jax.jit
def poloidal_extent_pure(
    gamma: jax.Array,
    gammadash: jax.Array,
    major_radius: float,
    z_position: float,
    theta_target: float,
    exponent: int,
) -> jax.Array:
    radius = jnp.linalg.norm(gamma[:, :2], axis=-1)
    theta_in = jnp.arctan2(gamma[:, 2] - z_position, -(radius - major_radius))
    arclength = jnp.linalg.norm(gammadash, axis=-1)
    excess = jnp.maximum(jnp.abs(theta_in) - theta_target, 0.0)
    return (1.0 / exponent) * jnp.mean((excess**exponent) * arclength)


@jax.jit
def projected_ellipse_width_pure(
    gamma: jax.Array,
    gammadash: jax.Array,
    major_radius: float,
    minor_radius: float,
    z_position: float,
    scale: float,
    epsilon: float,
) -> jax.Array:
    radius = jnp.linalg.norm(gamma[:, :2], axis=-1)
    phi = jnp.arctan2(gamma[:, 1], gamma[:, 0])
    theta = jnp.arctan2(gamma[:, 2] - z_position, -(radius - major_radius))
    phi_ref = jnp.arctan2(jnp.mean(jnp.sin(phi)), jnp.mean(jnp.cos(phi)))
    dphi = jnp.mod(phi - phi_ref + jnp.pi, 2.0 * jnp.pi) - jnp.pi
    projected = jnp.stack([major_radius * dphi, minor_radius * theta], axis=-1)
    dl = jnp.linalg.norm(gammadash, axis=-1)
    weights = dl / jnp.sum(dl)
    center = jnp.sum(weights[:, None] * projected, axis=0)
    centered = projected - center
    cov = (weights[:, None] * centered).T @ centered
    cov_xx = cov[0, 0]
    cov_xy = 0.5 * (cov[0, 1] + cov[1, 0])
    cov_yy = cov[1, 1]
    trace = cov_xx + cov_yy
    discriminant = jnp.maximum((cov_xx - cov_yy) ** 2 + 4.0 * cov_xy**2, 0.0)
    lambda_minor = 0.5 * (trace - jnp.sqrt(discriminant))
    return scale * jnp.sqrt(jnp.maximum(lambda_minor, epsilon))


def self_distance_pure(
    gamma: jax.Array,
    gammadash: jax.Array,
    minimum_distance: float,
    mask: jax.Array,
    normalize: bool,
) -> jax.Array:
    dist_sq = jnp.sum((gamma[:, None, :] - gamma[None, :, :]) ** 2, axis=2)
    safe_dist_sq = jnp.where(dist_sq > 0.0, dist_sq, 1.0)
    distances = jnp.where(dist_sq > 0.0, jnp.sqrt(safe_dist_sq), 0.0)
    arc_weights = (
        jnp.linalg.norm(gammadash, axis=1)[:, None]
        * jnp.linalg.norm(gammadash, axis=1)[None, :]
    )
    violation = jnp.maximum(minimum_distance - distances, 0.0) ** 2
    total = 0.5 * jnp.sum(mask * arc_weights * violation)
    if normalize:
        return total / (gamma.shape[0] ** 2)
    return total


class PoloidalExtentJAX(Optimizable):
    """JAX penalty for CWS banana-coil poloidal extent beyond a half-angle."""

    def __init__(
        self,
        curve,
        major_radius: float,
        theta_target: float,
        *,
        exponent: int = 4,
        z_position: float = 0.0,
    ):
        self.curve = curve
        self.major_radius = major_radius
        self.theta_target = theta_target
        self.exponent = exponent
        self.z_position = z_position
        self._grad_gamma = jax.jit(jax.grad(poloidal_extent_pure, argnums=0))
        self._grad_gammadash = jax.jit(jax.grad(poloidal_extent_pure, argnums=1))
        super().__init__(depends_on=[curve])

    def poloidal_half_width(self) -> float:
        gamma = np.asarray(self.curve.gamma(), dtype=float)
        radius = np.linalg.norm(gamma[:, :2], axis=-1)
        theta_in = np.arctan2(gamma[:, 2] - self.z_position, -(radius - self.major_radius))
        return float(np.max(np.abs(theta_in)))

    def J(self) -> float:
        return _as_float(
            poloidal_extent_pure(
                jnp.asarray(self.curve.gamma()),
                jnp.asarray(self.curve.gammadash()),
                self.major_radius,
                self.z_position,
                self.theta_target,
                self.exponent,
            )
        )

    @derivative_dec
    def dJ(self):
        gamma = jnp.asarray(self.curve.gamma())
        gammadash = jnp.asarray(self.curve.gammadash())
        grad_gamma = self._grad_gamma(
            gamma,
            gammadash,
            self.major_radius,
            self.z_position,
            self.theta_target,
            self.exponent,
        )
        grad_gammadash = self._grad_gammadash(
            gamma,
            gammadash,
            self.major_radius,
            self.z_position,
            self.theta_target,
            self.exponent,
        )
        return self.curve.dgamma_by_dcoeff_vjp(
            np.asarray(grad_gamma)
        ) + self.curve.dgammadash_by_dcoeff_vjp(np.asarray(grad_gammadash))

    return_fn_map = {"J": J, "dJ": dJ}


class ProjectedEllipseWidthJAX(Optimizable):
    """JAX scalar width of a CWS curve projected to winding-surface meters."""

    def __init__(
        self,
        curve,
        major_radius: float,
        minor_radius: float,
        *,
        z_position: float = 0.0,
        scale: float = 2.0 * np.sqrt(2.0),
        epsilon: float = 1.0e-20,
    ):
        self.curve = curve
        self.major_radius = major_radius
        self.minor_radius = minor_radius
        self.z_position = z_position
        self.scale = scale
        self.epsilon = epsilon
        self._grad_gamma = jax.jit(jax.grad(projected_ellipse_width_pure, argnums=0))
        self._grad_gammadash = jax.jit(
            jax.grad(projected_ellipse_width_pure, argnums=1)
        )
        super().__init__(depends_on=[curve])

    def J(self) -> float:
        return _as_float(
            projected_ellipse_width_pure(
                jnp.asarray(self.curve.gamma()),
                jnp.asarray(self.curve.gammadash()),
                self.major_radius,
                self.minor_radius,
                self.z_position,
                self.scale,
                self.epsilon,
            )
        )

    @derivative_dec
    def dJ(self):
        gamma = jnp.asarray(self.curve.gamma())
        gammadash = jnp.asarray(self.curve.gammadash())
        grad_gamma = self._grad_gamma(
            gamma,
            gammadash,
            self.major_radius,
            self.minor_radius,
            self.z_position,
            self.scale,
            self.epsilon,
        )
        grad_gammadash = self._grad_gammadash(
            gamma,
            gammadash,
            self.major_radius,
            self.minor_radius,
            self.z_position,
            self.scale,
            self.epsilon,
        )
        return self.curve.dgamma_by_dcoeff_vjp(
            np.asarray(grad_gamma)
        ) + self.curve.dgammadash_by_dcoeff_vjp(np.asarray(grad_gammadash))

    return_fn_map = {"J": J, "dJ": dJ}


class CurveSelfDistanceJAX(Optimizable):
    """JAX self-distance hinge for non-neighboring quadpoint pairs."""

    def __init__(
        self,
        curve,
        minimum_distance: float,
        *,
        neighbor_skip: int = 3,
        normalize: bool = False,
    ):
        self.curve = curve
        self.minimum_distance = minimum_distance
        self.neighbor_skip = neighbor_skip
        self.normalize = normalize
        point_count = len(curve.quadpoints)
        if not (0 <= neighbor_skip < point_count // 2):
            raise ValueError(
                f"neighbor_skip={neighbor_skip} must satisfy 0 <= neighbor_skip < "
                f"N/2 = {point_count // 2}"
            )
        indices = np.arange(point_count)
        distance = np.abs(indices[:, None] - indices[None, :])
        periodic_distance = np.minimum(distance, point_count - distance)
        self._mask_np = (periodic_distance > neighbor_skip).astype(np.float64)
        self._mask = jnp.asarray(self._mask_np)
        self._forward = jax.jit(self_distance_pure, static_argnums=4)
        self._grad_gamma = jax.jit(
            jax.grad(self_distance_pure, argnums=0),
            static_argnums=4,
        )
        self._grad_gammadash = jax.jit(
            jax.grad(self_distance_pure, argnums=1),
            static_argnums=4,
        )
        super().__init__(depends_on=[curve])

    def shortest_self_distance(self) -> float:
        gamma = np.asarray(self.curve.gamma(), dtype=float)
        distances = np.sqrt(
            np.sum((gamma[:, None, :] - gamma[None, :, :]) ** 2, axis=2)
        )
        masked = np.where(self._mask_np > 0.0, distances, np.inf)
        return float(np.min(masked))

    def J(self) -> float:
        return _as_float(
            self._forward(
                jnp.asarray(self.curve.gamma()),
                jnp.asarray(self.curve.gammadash()),
                self.minimum_distance,
                self._mask,
                self.normalize,
            )
        )

    @derivative_dec
    def dJ(self):
        gamma = jnp.asarray(self.curve.gamma())
        gammadash = jnp.asarray(self.curve.gammadash())
        grad_gamma = self._grad_gamma(
            gamma,
            gammadash,
            self.minimum_distance,
            self._mask,
            self.normalize,
        )
        grad_gammadash = self._grad_gammadash(
            gamma,
            gammadash,
            self.minimum_distance,
            self._mask,
            self.normalize,
        )
        return self.curve.dgamma_by_dcoeff_vjp(
            np.asarray(grad_gamma)
        ) + self.curve.dgammadash_by_dcoeff_vjp(np.asarray(grad_gammadash))

    return_fn_map = {"J": J, "dJ": dJ}


def weighted_sum_objective(terms: list[WeightedTerm]) -> Optimizable:
    active_terms = [term for term in terms if term.weight != 0.0]
    if not active_terms:
        raise ValueError("At least one objective term must have nonzero weight")
    objective = active_terms[0].weight * active_terms[0].objective
    for term in active_terms[1:]:
        objective = objective + term.weight * term.objective
    return objective


def add_current_penalties(
    terms: list[WeightedTerm],
    *,
    biotsavart: BiotSavart | BiotSavartJAX,
    weight: float,
) -> None:
    tf_current = biotsavart.coils[TF_IDX].current
    banana_current = biotsavart.coils[BANANA_IDX].current
    terms.append(
        WeightedTerm(
            "tf_current_max",
            weight,
            current_abs_upper_bound_penalty(
                tf_current,
                max(abs(limit) for limit in HARDWARE_LIMITS.tf_current_ka_limits)
                * KA_TO_A,
            ),
        )
    )
    terms.append(
        WeightedTerm(
            "banana_current_max",
            weight,
            current_abs_upper_bound_penalty(
                banana_current,
                max(abs(limit) for limit in HARDWARE_LIMITS.banana_current_ka_limits)
                * KA_TO_A,
            ),
        )
    )


def banana_geometry_terms(
    *,
    biotsavart: BiotSavart | BiotSavartJAX,
    surface: SurfaceRZFourier | SurfaceXYZFourier | SurfaceXYZTensorFourier | None,
    length_weight: float,
    ccdist_weight: float,
    csdist_weight: float,
    curvature_weight: float,
    poloidal_weight: float,
    width_weight: float,
    self_distance_weight: float,
    current_weight: float,
    include_current_penalties: bool,
    include_min_length: bool,
    include_width: bool,
) -> list[WeightedTerm]:
    curves = banana_curves(biotsavart)
    base_curve = banana_base_curve(biotsavart)
    terms: list[WeightedTerm] = [
        WeightedTerm(
            "length_max",
            length_weight,
            QuadraticPenalty(
                CurveLengthJAX(base_curve),
                HARDWARE_LIMITS.max_length,
                "max",
            ),
        ),
        WeightedTerm(
            "coil_coil_distance",
            ccdist_weight,
            CurveCurveDistanceJAX(curves, HARDWARE_LIMITS.min_ccdist),
        ),
        WeightedTerm(
            "curvature",
            curvature_weight,
            LpCurveCurvatureJAX(
                base_curve,
                HARDWARE_LIMITS.banana_curv_p,
                HARDWARE_LIMITS.max_curvature,
            ),
        ),
        WeightedTerm(
            "poloidal_extent",
            poloidal_weight,
            PoloidalExtentJAX(
                base_curve,
                HBT_BANANA_WS.major_radius,
                np.deg2rad(BANANA_TARGETS.poloidal_deg),
                exponent=HARDWARE_LIMITS.banana_curv_p,
            ),
        ),
        WeightedTerm(
            "self_distance",
            self_distance_weight,
            CurveSelfDistanceJAX(
                base_curve,
                BANANA_TARGETS.self_distance_min,
                neighbor_skip=max(3, int(1.5 * getattr(base_curve, "order", 3))),
            ),
        ),
    ]
    if include_min_length:
        terms.append(
            WeightedTerm(
                "length_min",
                length_weight,
                QuadraticPenalty(
                    CurveLengthJAX(base_curve),
                    0.5 * HARDWARE_LIMITS.max_length,
                    "min",
                ),
            )
        )
    if surface is not None and csdist_weight != 0.0:
        terms.append(
            WeightedTerm(
                "coil_surface_distance",
                csdist_weight,
                CurveSurfaceDistanceJAX(curves, surface, HARDWARE_LIMITS.min_csdist),
            )
        )
    if include_width:
        width = ProjectedEllipseWidthJAX(
            base_curve,
            HBT_BANANA_WS.major_radius,
            HBT_BANANA_WS.minor_radius,
        )
        terms.append(
            WeightedTerm(
                "width_max",
                width_weight,
                QuadraticPenalty(width, BANANA_TARGETS.width_max, "max"),
            )
        )
        terms.append(
            WeightedTerm(
                "width_min",
                width_weight,
                QuadraticPenalty(width, BANANA_TARGETS.width_min, "min"),
            )
        )
    if include_current_penalties:
        add_current_penalties(terms, biotsavart=biotsavart, weight=current_weight)
    return terms


def build_stage2_objective(
    *,
    biotsavart_jax: BiotSavartJAX,
    surface: SurfaceRZFourier | SurfaceXYZFourier | SurfaceXYZTensorFourier,
    weights: Stage2Weights,
    include_current_penalties: bool,
    include_min_length: bool,
    include_width: bool,
) -> tuple[Optimizable, list[WeightedTerm]]:
    terms = [
        WeightedTerm(
            "squared_flux",
            weights.sqflux,
            SquaredFluxJAX(surface, biotsavart_jax, definition="normalized"),
        )
    ]
    terms.extend(
        banana_geometry_terms(
            biotsavart=biotsavart_jax,
            surface=surface,
            length_weight=weights.length,
            ccdist_weight=weights.ccdist,
            csdist_weight=weights.csdist,
            curvature_weight=weights.curvature,
            poloidal_weight=weights.poloidal,
            width_weight=weights.width,
            self_distance_weight=weights.selfint,
            current_weight=weights.currents,
            include_current_penalties=include_current_penalties,
            include_min_length=include_min_length,
            include_width=include_width,
        )
    )
    return weighted_sum_objective(terms), terms


def build_single_stage_objective(
    *,
    boozersurface: BoozerSurfaceJAX,
    biotsavart_jax: BiotSavartJAX,
    iota_target: float,
    weights: SingleStageWeights,
    include_boozer_residual: bool,
    include_current_penalties: bool,
    include_min_length: bool,
    include_width: bool,
) -> tuple[Optimizable, list[WeightedTerm]]:
    terms = [
        WeightedTerm(
            "non_quasisymmetric_ratio",
            weights.nonqs,
            NonQuasiSymmetricRatioJAX(boozersurface, biotsavart_jax),
        ),
        WeightedTerm(
            "iota",
            weights.iota,
            QuadraticPenalty(IotasJAX(boozersurface), iota_target),
        ),
    ]
    if include_boozer_residual:
        terms.append(
            WeightedTerm(
                "boozer_residual",
                weights.bres,
                BoozerResidualJAX(boozersurface, biotsavart_jax),
            )
        )
    terms.extend(
        banana_geometry_terms(
            biotsavart=biotsavart_jax,
            surface=boozersurface.surface,
            length_weight=weights.length,
            ccdist_weight=weights.ccdist,
            csdist_weight=weights.csdist,
            curvature_weight=weights.curvature,
            poloidal_weight=weights.poloidal,
            width_weight=weights.width,
            self_distance_weight=weights.selfint,
            current_weight=weights.currents,
            include_current_penalties=include_current_penalties,
            include_min_length=include_min_length,
            include_width=include_width,
        )
    )
    return weighted_sum_objective(terms), terms


def diagnostics_from_terms(objective: Optimizable, terms: list[WeightedTerm]) -> dict[str, float]:
    diagnostics = {
        "J": float(objective.J()),
        "grad_norm": float(np.linalg.norm(np.asarray(objective.dJ(), dtype=float))),
    }
    for term in terms:
        value = float(term.objective.J())
        diagnostics[f"J_{term.name}"] = value
        diagnostics[f"weighted_{term.name}"] = float(term.weight) * value
    return diagnostics


def banana_geometry_diagnostics(biotsavart: BiotSavart | BiotSavartJAX) -> dict[str, float]:
    base_curve = banana_base_curve(biotsavart)
    poloidal = PoloidalExtentJAX(
        base_curve,
        HBT_BANANA_WS.major_radius,
        np.deg2rad(BANANA_TARGETS.poloidal_deg),
    )
    width = ProjectedEllipseWidthJAX(
        base_curve,
        HBT_BANANA_WS.major_radius,
        HBT_BANANA_WS.minor_radius,
    )
    self_distance = CurveSelfDistanceJAX(
        base_curve,
        BANANA_TARGETS.self_distance_min,
        neighbor_skip=max(3, int(1.5 * getattr(base_curve, "order", 3))),
    )
    return {
        "coil_length_m": float(CurveLengthJAX(base_curve).J()),
        "max_curvature_inv_m": float(np.max(np.asarray(base_curve.kappa(), dtype=float))),
        "poloidal_half_width_rad": poloidal.poloidal_half_width(),
        "projected_width_m": float(width.J()),
        "self_distance_m": self_distance.shortest_self_distance(),
        "tf_current_A": float(biotsavart.coils[TF_IDX].current.get_value()),
        "banana_current_A": float(biotsavart.coils[BANANA_IDX].current.get_value()),
    }


def minimize_soft_penalty(
    *,
    objective: Optimizable,
    maxiter: int,
    log: DriverLog,
) -> tuple[object, EvaluationTracker]:
    started_at = time.monotonic()
    evaluations = 0
    iterations = 0

    def value_and_gradient(dofs: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal evaluations
        objective.x = dofs
        value = float(objective.J())
        gradient = np.asarray(objective.dJ(), dtype=float)
        evaluations += 1
        log(
            "eval="
            f"{evaluations}, iter={iterations}, "
            f"J={value:.16e}, |grad|={np.linalg.norm(gradient):.16e}"
        )
        return value, gradient

    def callback(dofs: np.ndarray) -> None:
        nonlocal iterations
        objective.x = dofs
        iterations += 1
        log(f"accepted_iter={iterations}, J={float(objective.J()):.16e}")

    result = minimize(
        value_and_gradient,
        np.asarray(objective.x, dtype=float),
        jac=True,
        method="L-BFGS-B",
        callback=callback,
        tol=1.0e-15,
        options={"maxiter": int(maxiter), "maxcor": 300},
    )
    objective.x = np.asarray(result.x, dtype=float)
    return result, EvaluationTracker(
        iterations=iterations,
        evaluations=evaluations,
        started_at=started_at,
    )


def require_successful_boozer_solve(
    boozersurface: BoozerSurfaceJAX,
    state: BoozerSolveState,
) -> BoozerSolveState:
    result = boozersurface.run_code(state.iota, state.G)
    if not bool(result["success"]):
        raise RuntimeError(
            "Boozer solve failed during JAX banana single-stage optimization"
        )
    return BoozerSolveState(
        iota=float(result["iota"]),
        G=float(result["G"]),
    )


def minimize_single_stage_soft_penalty(
    *,
    objective: Optimizable,
    boozersurface: BoozerSurfaceJAX,
    initial_state: BoozerSolveState,
    maxiter: int,
    log: DriverLog,
) -> tuple[object, EvaluationTracker]:
    started_at = time.monotonic()
    evaluations = 0
    iterations = 0
    solve_state = initial_state

    def value_and_gradient(dofs: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal evaluations, solve_state
        objective.x = dofs
        solve_state = require_successful_boozer_solve(boozersurface, solve_state)
        value = float(objective.J())
        gradient = np.asarray(objective.dJ(), dtype=float)
        evaluations += 1
        log(
            "eval="
            f"{evaluations}, iter={iterations}, "
            f"iota={solve_state.iota:.16e}, J={value:.16e}, "
            f"|grad|={np.linalg.norm(gradient):.16e}"
        )
        return value, gradient

    def callback(dofs: np.ndarray) -> None:
        nonlocal iterations
        objective.x = dofs
        iterations += 1
        log(f"accepted_iter={iterations}, J={float(objective.J()):.16e}")

    result = minimize(
        value_and_gradient,
        np.asarray(objective.x, dtype=float),
        jac=True,
        method="L-BFGS-B",
        callback=callback,
        tol=1.0e-15,
        options={"maxiter": int(maxiter), "maxcor": 300},
    )
    objective.x = np.asarray(result.x, dtype=float)
    require_successful_boozer_solve(boozersurface, solve_state)
    return result, EvaluationTracker(
        iterations=iterations,
        evaluations=evaluations,
        started_at=started_at,
    )


def build_jax_boozer_surface(
    *,
    biotsavart_jax: BiotSavartJAX,
    surface: SurfaceRZFourier | SurfaceXYZFourier | SurfaceXYZTensorFourier,
    mpol: int,
    ntor: int,
    nphi: int,
    ntheta: int,
    constraint_weight: float | None,
) -> BoozerSurfaceJAX:
    boozer_surface = build_boozer_surface_copy(
        surface,
        mpol=mpol,
        ntor=ntor,
        nphi=nphi,
        ntheta=ntheta,
    )
    label = Volume(boozer_surface)
    return BoozerSurfaceJAX(
        biotsavart_jax,
        boozer_surface,
        label,
        boozer_surface.volume(),
        constraint_weight=constraint_weight,
        options={"verbose": False},
    )


def save_chainable_boozer_surface(
    path: Path,
    *,
    biotsavart: BiotSavart,
    surface: SurfaceRZFourier | SurfaceXYZFourier | SurfaceXYZTensorFourier,
    constraint_weight: float | None,
) -> None:
    boozer_surface = surface
    if not isinstance(boozer_surface, SurfaceXYZFourier | SurfaceXYZTensorFourier):
        boozer_surface = build_boozer_surface_copy(
            surface,
            mpol=surface.mpol,
            ntor=surface.ntor,
            nphi=len(surface.quadpoints_phi),
            ntheta=len(surface.quadpoints_theta),
        )
    label = Volume(boozer_surface)
    booz = BoozerSurface(
        biotsavart,
        boozer_surface,
        label,
        boozer_surface.volume(),
        constraint_weight=constraint_weight,
        options={"verbose": False},
    )
    booz.save(str(path))


def load_stage2_boozer_seed(
    path: str | Path,
) -> tuple[BiotSavart, SurfaceRZFourier | SurfaceXYZFourier | SurfaceXYZTensorFourier]:
    booz = load(str(path))
    if not isinstance(booz, BoozerSurface):
        raise TypeError(f"Expected BoozerSurface JSON, got {type(booz)!r}")
    return booz.biotsavart, booz.surface


def result_payload(
    *,
    driver: str,
    args: dict[str, object],
    optimizer_result: object,
    tracker: EvaluationTracker,
    diagnostics: dict[str, float],
    paths: BananaDriverPaths,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "driver": driver,
        "args": args,
        "optimizer": {
            "success": bool(optimizer_result.success),
            "message": str(optimizer_result.message),
            "nit": int(optimizer_result.nit),
            "nfev": int(optimizer_result.nfev),
            "fun": float(optimizer_result.fun),
        },
        "tracker": asdict(tracker),
        "hardware": {
            "winding_surface": asdict(HBT_BANANA_WS),
            "limits": asdict(HARDWARE_LIMITS),
            "targets": asdict(BANANA_TARGETS),
        },
        "diagnostics": diagnostics,
        "outputs": {
            "biotsavart": str(paths.biotsavart),
            "boozersurface": str(paths.boozersurface),
            "surface": None if paths.surface is None else str(paths.surface),
            "log": str(paths.log),
        },
    }
