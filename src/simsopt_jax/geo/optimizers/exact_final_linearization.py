"""Private device-pytree contract for one exact final-state linearization."""

from __future__ import annotations

from dataclasses import dataclass, field

import jax

_CONSTRUCTION_TOKEN = object()


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, slots=True)
class _ExactFinalLinearizationInputs:
    """Atomic solved-state and coil inputs consumed by one linearization."""

    solved_state: jax.Array
    coil_dofs: jax.Array
    coil_dynamic_inputs: tuple[jax.Array, ...]
    residual_configuration: tuple[jax.Array, ...]
    _construction_token: object = field(
        init=False,
        repr=False,
        compare=False,
        default=None,
    )

    def __post_init__(self) -> None:
        if self._construction_token is not _CONSTRUCTION_TOKEN:
            raise RuntimeError(
                "Exact final-linearization inputs must be minted atomically."
            )

    def tree_flatten(self):
        return (
            self.solved_state,
            self.coil_dofs,
            self.coil_dynamic_inputs,
            self.residual_configuration,
        ), None

    @classmethod
    def _mint(
        cls,
        *,
        solved_state,
        coil_dofs,
        coil_dynamic_inputs,
        residual_configuration,
    ):
        instance = object.__new__(cls)
        object.__setattr__(instance, "solved_state", solved_state)
        object.__setattr__(instance, "coil_dofs", coil_dofs)
        object.__setattr__(instance, "coil_dynamic_inputs", coil_dynamic_inputs)
        object.__setattr__(
            instance,
            "residual_configuration",
            residual_configuration,
        )
        object.__setattr__(instance, "_construction_token", _CONSTRUCTION_TOKEN)
        return instance

    @classmethod
    def tree_unflatten(cls, _aux_data, children):
        solved_state, coil_dofs, coil_dynamic_inputs, residual_configuration = children
        return cls._mint(
            solved_state=solved_state,
            coil_dofs=coil_dofs,
            coil_dynamic_inputs=coil_dynamic_inputs,
            residual_configuration=residual_configuration,
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, slots=True)
class _ExactFinalLinearizationIdentity:
    """Exact device snapshot of the fields that define one linearization."""

    inputs: _ExactFinalLinearizationInputs
    residual: jax.Array
    jacobian: jax.Array
    orientation_code: jax.Array
    producer_solve_success: jax.Array
    _construction_token: object = field(
        init=False,
        repr=False,
        compare=False,
        default=None,
    )

    def __post_init__(self) -> None:
        if self._construction_token is not _CONSTRUCTION_TOKEN:
            raise RuntimeError(
                "Exact final-linearization identities must be minted atomically."
            )

    def tree_flatten(self):
        return (
            self.inputs,
            self.residual,
            self.jacobian,
            self.orientation_code,
            self.producer_solve_success,
        ), None

    @classmethod
    def _mint(
        cls,
        *,
        inputs,
        residual,
        jacobian,
        orientation_code,
        producer_solve_success,
    ):
        instance = object.__new__(cls)
        object.__setattr__(instance, "inputs", inputs)
        object.__setattr__(instance, "residual", residual)
        object.__setattr__(instance, "jacobian", jacobian)
        object.__setattr__(instance, "orientation_code", orientation_code)
        object.__setattr__(
            instance,
            "producer_solve_success",
            producer_solve_success,
        )
        object.__setattr__(instance, "_construction_token", _CONSTRUCTION_TOKEN)
        return instance

    @classmethod
    def tree_unflatten(cls, _aux_data, children):
        inputs, residual, jacobian, orientation_code, producer_solve_success = children
        return cls._mint(
            inputs=inputs,
            residual=residual,
            jacobian=jacobian,
            orientation_code=orientation_code,
            producer_solve_success=producer_solve_success,
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, slots=True)
class _ExactFinalLinearization:
    """Device-only final-state Jacobian and factors bound to atomic inputs."""

    inputs: _ExactFinalLinearizationInputs
    residual: jax.Array
    jacobian: jax.Array
    lu: jax.Array
    pivots: jax.Array
    orientation_code: jax.Array
    producer_solve_success: jax.Array
    identity: _ExactFinalLinearizationIdentity
    _construction_token: object = field(
        init=False,
        repr=False,
        compare=False,
        default=None,
    )

    def __post_init__(self) -> None:
        if self._construction_token is not _CONSTRUCTION_TOKEN:
            raise RuntimeError(
                "Exact final linearizations must be minted by their producer."
            )

    def tree_flatten(self):
        return (
            self.inputs,
            self.residual,
            self.jacobian,
            self.lu,
            self.pivots,
            self.orientation_code,
            self.producer_solve_success,
            self.identity,
        ), None

    @classmethod
    def _mint(
        cls,
        *,
        inputs,
        residual,
        jacobian,
        lu,
        pivots,
        orientation_code,
        producer_solve_success,
        identity,
    ):
        instance = object.__new__(cls)
        object.__setattr__(instance, "inputs", inputs)
        object.__setattr__(instance, "residual", residual)
        object.__setattr__(instance, "jacobian", jacobian)
        object.__setattr__(instance, "lu", lu)
        object.__setattr__(instance, "pivots", pivots)
        object.__setattr__(instance, "orientation_code", orientation_code)
        object.__setattr__(
            instance,
            "producer_solve_success",
            producer_solve_success,
        )
        object.__setattr__(instance, "identity", identity)
        object.__setattr__(instance, "_construction_token", _CONSTRUCTION_TOKEN)
        return instance

    @classmethod
    def tree_unflatten(cls, _aux_data, children):
        (
            inputs,
            residual,
            jacobian,
            lu,
            pivots,
            orientation_code,
            producer_solve_success,
            identity,
        ) = children
        return cls._mint(
            inputs=inputs,
            residual=residual,
            jacobian=jacobian,
            lu=lu,
            pivots=pivots,
            orientation_code=orientation_code,
            producer_solve_success=producer_solve_success,
            identity=identity,
        )


def _mint_exact_final_linearization_inputs(
    *,
    solved_state: jax.Array,
    coil_dofs: jax.Array,
    coil_dynamic_inputs: tuple[jax.Array, ...],
    residual_configuration: tuple[jax.Array, ...],
) -> _ExactFinalLinearizationInputs:
    return _ExactFinalLinearizationInputs._mint(
        solved_state=solved_state,
        coil_dofs=coil_dofs,
        coil_dynamic_inputs=coil_dynamic_inputs,
        residual_configuration=residual_configuration,
    )


def _mint_exact_final_linearization(
    *,
    inputs: _ExactFinalLinearizationInputs,
    residual: jax.Array,
    jacobian: jax.Array,
    lu: jax.Array,
    pivots: jax.Array,
    orientation_code: jax.Array,
    producer_solve_success: jax.Array,
    identity: _ExactFinalLinearizationIdentity,
) -> _ExactFinalLinearization:
    return _ExactFinalLinearization._mint(
        inputs=inputs,
        residual=residual,
        jacobian=jacobian,
        lu=lu,
        pivots=pivots,
        orientation_code=orientation_code,
        producer_solve_success=producer_solve_success,
        identity=identity,
    )


def _mint_exact_final_linearization_identity(
    *,
    inputs: _ExactFinalLinearizationInputs,
    residual: jax.Array,
    jacobian: jax.Array,
    orientation_code: jax.Array,
    producer_solve_success: jax.Array,
) -> _ExactFinalLinearizationIdentity:
    return _ExactFinalLinearizationIdentity._mint(
        inputs=inputs,
        residual=residual,
        jacobian=jacobian,
        orientation_code=orientation_code,
        producer_solve_success=producer_solve_success,
    )
