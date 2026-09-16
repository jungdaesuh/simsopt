"""One construction that reads a flat-675 vector as the nested route's problem.

The flat formulation and the nested route disagree about almost everything at
the surface: the flat route carries all 675 coordinates in one vector and
closes ``(iota, G)`` by a two-column least-squares solve inside the forward
pass, while the nested route carries a mutable ``BoozerSurface`` whose inner
Newton drives a reduced gradient to tolerance.  Comparing an endpoint across
the two is only meaningful if both are looking at *the same problem*, so this
module owns the single translation and nothing else.

Two things make that translation non-trivial, and both are load-bearing:

1. **The coil source.** The flat-675 material's field is a function of its
   ``CoilSetDofExtractionSpec`` and the vector's coil block — nothing else.
   A frozen bundle also ships a ``native_biot_savart.json``, which is a
   *second, independent* record of the same coils, and the two have been
   observed to disagree: in the ``single_stage_seed_iota15`` fixture they
   differ by 0.583 in the owner DOFs and by a fifth in ``B`` on the surface, so
   the JAX lane and the native lane there solve different Boozer problems.
   This module therefore reconstructs simsopt ``Coil`` objects from the
   material's own extraction spec, by inverting the same map the JAX kernels
   apply, and never reads an archived Biot-Savart file — whether or not the
   archive happens to agree on a given bundle.
2. **The quadrature.** The nested residual is evaluated on the surface's own
   quadrature, so the native surface must carry the material's
   ``quadpoints_phi``/``quadpoints_theta`` and resolution, not a fresh default.

The record this module returns is immutable, and every mutable solver object a
consumer needs is built fresh from it on request: the reduced nested-LS entry
points mutate ``jax_boozer.surface`` in place, so sharing one instance across
two lanes would make the comparison order-dependent.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
from simsopt.field import BiotSavart
from simsopt.field.coil import Coil, Current, ScaledCurrent
from simsopt.geo import (
    CurveXYZFourier,
    RotatedCurve,
    SurfaceRZFourier,
    SurfaceXYZTensorFourier,
    Volume,
)
from simsopt_jax.core.curve_geometry import optimizable_input_dofs_from_map_spec
from simsopt_jax.core.specs import (
    CoilDofExtractionSpec,
    CoilSetDofExtractionSpec,
    CurveCWSFourierRZSpec,
    CurveXYZFourierSpec,
    OptimizableDofMapSpec,
    SurfaceXYZTensorFourierSpec,
)
from simsopt_jax.core.surface_rzfourier import surface_rz_fourier_dofs_from_spec

from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.geo.boozer_surface import BoozerSurfaceJAX
from simsopt_jax_adapters.geo.curvecwsfourier import CurveCWSFourier
from simsopt_jax_adapters.geo.nested_ls_contract import (
    NESTED_LS_CONSTRAINT_WEIGHT,
    NESTED_LS_BANANA_NEWTON_MAXITER,
    NESTED_LS_BANANA_NEWTON_TOL,
    NESTED_LS_WEIGHT_INV_MODB,
)

from .boozer_material import build_flat675_boozer_system, flat675_candidate_geometry
from .construction import Flat675Problem
from .layout import FlatSingleStageLayout
from .policy import Flat675BoozerLabelType
from .y_solve import solve_flat675_y_qr

# ``RotatedCurve`` builds its matrix from ``(phi, flip)`` and the extraction
# spec stores only the product, so the bridge recovers the pair and then
# CHECKS the reconstruction against the stored matrix.  The tolerance is the
# round-trip error of one ``arctan2`` plus one ``cos``/``sin`` pair, not a
# physics tolerance: anything larger means the stored matrix is not a
# ``RotatedCurve`` matrix at all, which is a different coil set.
_ROTMAT_RECONSTRUCTION_ATOL: float = 1.0e-14

#: The nested route's inner-solver options at the TIMING bar -- the banana
#: ``run_code`` policy (``NESTED_LS_BANANA_NEWTON_TOL`` /
#: ``NESTED_LS_BANANA_NEWTON_MAXITER``), which is the policy both lanes of the
#: endpoint comparison are judged at, and NOT the tighter reconstruct/physics
#: bar ``NESTED_LS_NEWTON_TOL``.  These are the nested contract's options,
#: deliberately not the flat-675 objective's: the question this bridge exists
#: to answer is what the NESTED route would say about the flat route's
#: surface.  Public because the harness tests pin these values instead of
#: restating them, which is how the two drift apart.
NESTED_BRIDGE_NEWTON_OPTIONS: MappingProxyType = MappingProxyType(
    {
        "verbose": False,
        "newton_tol": NESTED_LS_BANANA_NEWTON_TOL,
        "newton_maxiter": NESTED_LS_BANANA_NEWTON_MAXITER,
        "weight_inv_modB": NESTED_LS_WEIGHT_INV_MODB,
        "optimizer_backend": "scipy",
        "materialize_dense_linearization": False,
    }
)


class Flat675NestedBridgeError(ValueError):
    """Raised when a flat-675 point cannot be read as the nested problem."""


def _host_float64(values: object) -> NDArray[np.float64]:
    return np.array(jax.device_get(values), dtype=np.float64, copy=True)


def _mapped_dofs(
    map_spec: OptimizableDofMapSpec,
    coil_dofs: jax.Array,
) -> NDArray[np.float64]:
    """The owner-DOF map the JAX kernels apply, evaluated on the host.

    This is the same function :mod:`simsopt_jax.core.field` calls to build the
    traced coil specs, so a native curve built from its result carries the
    coordinates the flat-675 field actually used.
    """
    return _host_float64(
        optimizable_input_dofs_from_map_spec(map_spec, coil_dofs)
    ).reshape(-1)


def _native_surface(
    template: SurfaceXYZTensorFourierSpec,
    surface_dofs: NDArray[np.float64],
) -> SurfaceXYZTensorFourier:
    """A native surface on the material's own resolution and quadrature."""
    surface = SurfaceXYZTensorFourier(
        mpol=int(template.mpol),
        ntor=int(template.ntor),
        nfp=int(template.nfp),
        stellsym=bool(template.stellsym),
        quadpoints_phi=_host_float64(template.quadpoints_phi),
        quadpoints_theta=_host_float64(template.quadpoints_theta),
    )
    surface.set_dofs(np.asarray(surface_dofs, dtype=np.float64))
    return surface


def _native_curve(curve_spec: object, full_dofs: NDArray[np.float64]):
    """The simsopt curve one curve spec plus its mapped DOF block names."""
    if isinstance(curve_spec, CurveXYZFourierSpec):
        curve = CurveXYZFourier(
            _host_float64(curve_spec.quadpoints),
            int(curve_spec.order),
        )
        curve.local_full_x = full_dofs
        return curve
    if isinstance(curve_spec, CurveCWSFourierRZSpec):
        winding = curve_spec.surface
        surface = SurfaceRZFourier(
            nfp=int(winding.nfp),
            stellsym=bool(winding.stellsym),
            mpol=int(winding.mpol),
            ntor=int(winding.ntor),
            quadpoints_phi=_host_float64(winding.quadpoints_phi),
            quadpoints_theta=_host_float64(winding.quadpoints_theta),
        )
        surface.set_dofs(_host_float64(surface_rz_fourier_dofs_from_spec(winding)))
        curve = CurveCWSFourier(
            quadpoints=_host_float64(curve_spec.quadpoints),
            order=int(curve_spec.order),
            surf=surface,
            G=float(curve_spec.G),
            H=float(curve_spec.H),
        )
        curve.local_full_x = full_dofs
        return curve
    raise Flat675NestedBridgeError(
        "the flat-675 coil extraction names a curve family this bridge cannot "
        f"reconstruct natively: {type(curve_spec).__name__}. Add it here rather "
        "than substituting a different coil set."
    )


def _rotated_curve(curve: object, rotmat: object) -> RotatedCurve:
    """Invert ``RotatedCurve``'s ``(phi, flip)`` -> matrix map, then verify it."""
    matrix = _host_float64(rotmat)
    flip = bool(matrix[2, 2] < 0.0)
    sin_phi = -matrix[0, 1] if flip else matrix[0, 1]
    rotated = RotatedCurve(curve, float(np.arctan2(sin_phi, matrix[0, 0])), flip)
    error = float(np.max(np.abs(np.asarray(rotated.rotmat) - matrix)))
    if error > _ROTMAT_RECONSTRUCTION_ATOL:
        raise Flat675NestedBridgeError(
            "the flat-675 coil extraction carries a symmetry matrix that is "
            "not a RotatedCurve(phi, flip) matrix; the native reconstruction "
            f"differs by {error!r}. Reconstructing anyway would place a coil "
            "somewhere the flat-675 field never put it."
        )
    return rotated


def _native_coil(
    extraction: CoilDofExtractionSpec,
    coil_dofs: jax.Array,
    curves_by_source: dict[int, object],
) -> Coil:
    if extraction.surface_map is not None or extraction.current_term_maps:
        raise Flat675NestedBridgeError(
            "this bridge reconstructs coils whose curve and scalar current are "
            "the only owner-DOF consumers; the extraction spec carries a "
            "surface map or affine current terms, which a native "
            "reconstruction would silently drop."
        )
    source_index = extraction.curve_source_index
    base = None if source_index is None else curves_by_source.get(source_index)
    if base is None:
        base = _native_curve(
            extraction.curve,
            _mapped_dofs(extraction.curve_map, coil_dofs),
        )
        if source_index is not None:
            curves_by_source[source_index] = base
    symmetry = extraction.symmetry
    curve = _rotated_curve(base, symmetry.rotmat) if symmetry.has_rotation else base
    current_dofs = _mapped_dofs(extraction.current_map, coil_dofs)
    if current_dofs.shape != (1,):
        raise Flat675NestedBridgeError(
            "flat-675 coil currents must be scalar Current DOFs; got shape "
            f"{current_dofs.shape}."
        )
    current = Current(float(current_dofs[0]))
    scale = float(symmetry.scale)
    return Coil(curve, current if scale == 1.0 else ScaledCurrent(current, scale))


def native_coils_from_flat675_material(
    extraction: CoilSetDofExtractionSpec,
    coil_dofs: NDArray[np.float64],
) -> tuple[Coil, ...]:
    """Native coils reproducing the flat-675 material's coil set exactly.

    The only inputs are the material's own extraction spec and the vector's
    coil block, which is the whole point: the archived
    ``native_biot_savart.json`` is a different coil set and is never consulted.
    """
    dofs = jnp.asarray(np.asarray(coil_dofs, dtype=np.float64), dtype=jnp.float64)
    curves_by_source: dict[int, object] = {}
    return tuple(
        _native_coil(coil, dofs, curves_by_source) for coil in extraction.coils
    )


@dataclass(frozen=True, slots=True)
class NestedJaxInputs:
    """Everything the reduced nested-LS entry points need, minus solver state.

    ``new_boozer_surface_jax`` is a factory rather than a stored instance
    because :func:`~simsopt_jax_adapters.geo.nested_ls_reduced.run_reduced_nested_ls_newton`
    and its Schur sibling mutate ``jax_boozer.surface`` in place.  One instance
    shared between a JAX lane and a native lane would mean the second lane
    starts from the first lane's answer.
    """

    biotsavart_jax: BiotSavartJAX
    surface_template: SurfaceXYZTensorFourierSpec
    surface_dofs: NDArray[np.float64]
    label_target: float
    constraint_weight: float
    weight_inv_modB: bool
    newton_options: MappingProxyType

    def new_boozer_surface_jax(self) -> BoozerSurfaceJAX:
        """A fresh solver object at the incoming surface, safe to mutate."""
        surface = _native_surface(self.surface_template, self.surface_dofs)
        return BoozerSurfaceJAX(
            self.biotsavart_jax,
            surface,
            Volume(surface),
            self.label_target,
            constraint_weight=self.constraint_weight,
            options=dict(self.newton_options),
        )


@dataclass(frozen=True, slots=True)
class NestedView:
    """One flat-675 vector, read as the nested route's problem objects."""

    coil_dofs: NDArray[np.float64]
    vessel_dofs: NDArray[np.float64]
    surface_dofs: NDArray[np.float64]
    iota: float
    G: float
    coils_native: tuple[Coil, ...]
    biotsavart_native: BiotSavart
    surface_native: SurfaceXYZTensorFourier
    label_native: Volume
    jax_inputs: NestedJaxInputs
    layout: FlatSingleStageLayout
    vector: NDArray[np.float64]


def nested_view_from_flat675(
    problem: Flat675Problem,
    vector: NDArray[np.float64],
) -> NestedView:
    """Read one flat-675 outer vector as the nested route's problem objects.

    ``(iota, G)`` come from the flat route's own two-column QR solve at this
    vector, evaluated through the same three calls the objective makes, so the
    nested solve starts at the flat route's inner state rather than at a
    re-derived one.
    """
    layout = problem.material.layout
    coordinates = np.asarray(vector, dtype=np.float64)
    if coordinates.shape != (layout.outer_dof_count,):
        raise Flat675NestedBridgeError(
            "a flat-675 outer vector must have shape "
            f"({layout.outer_dof_count},); got {coordinates.shape}."
        )
    if problem.objective_policy.boozer_label_type is not Flat675BoozerLabelType.VOLUME:
        raise Flat675NestedBridgeError(
            "the nested route's certified rows constrain the Volume label; this "
            "problem constrains "
            f"{problem.objective_policy.boozer_label_type.value!r}."
        )

    coil_dofs = np.array(coordinates[layout.coil_slice], copy=True)
    vessel_dofs = np.array(coordinates[layout.vessel_slice], copy=True)
    surface_dofs = np.array(coordinates[layout.surface_slice], copy=True)

    material = problem.material.boozer
    geometry = flat675_candidate_geometry(
        material,
        jnp.asarray(coil_dofs, dtype=jnp.float64),
        jnp.asarray(surface_dofs, dtype=jnp.float64),
    )
    system = build_flat675_boozer_system(geometry, problem.boozer_policy)
    inner_state = _host_float64(
        solve_flat675_y_qr(system.design_matrix, system.right_hand_side).solution
    )

    coils = native_coils_from_flat675_material(material.coil_dof_extraction, coil_dofs)
    surface = _native_surface(material.surface_template, surface_dofs)
    return NestedView(
        coil_dofs=coil_dofs,
        vessel_dofs=vessel_dofs,
        surface_dofs=surface_dofs,
        iota=float(inner_state[0]),
        G=float(inner_state[1]),
        coils_native=coils,
        biotsavart_native=BiotSavart(list(coils)),
        surface_native=surface,
        label_native=Volume(surface),
        jax_inputs=NestedJaxInputs(
            biotsavart_jax=BiotSavartJAX(list(coils)),
            surface_template=material.surface_template,
            surface_dofs=surface_dofs,
            label_target=float(problem.objective_policy.boozer_target_label),
            constraint_weight=NESTED_LS_CONSTRAINT_WEIGHT,
            weight_inv_modB=NESTED_LS_WEIGHT_INV_MODB,
            newton_options=NESTED_BRIDGE_NEWTON_OPTIONS,
        ),
        layout=layout,
        vector=np.array(coordinates, copy=True),
    )


__all__ = [
    "NESTED_BRIDGE_NEWTON_OPTIONS",
    "Flat675NestedBridgeError",
    "NestedJaxInputs",
    "NestedView",
    "native_coils_from_flat675_material",
    "nested_view_from_flat675",
]
