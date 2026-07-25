"""Typed device-to-host evaluation records for SciPy-controlled optimizers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Protocol, TypeVar, runtime_checkable

import jax
import numpy as np

from ._evaluation_lifecycle import (
    TargetScipyEvaluationClass,
    TargetScipyEvaluationLifecycle,
    TargetScipyEvaluationOutcome,
)


__all__ = [
    "TargetScipyDeviceIdentity",
    "TargetScipyDeviceEvaluation",
    "TargetScipyDevicePacketLayout",
    "TargetScipyEvaluationClass",
    "TargetScipyEvaluationLifecycle",
    "TargetScipyEvaluationOutcome",
    "TargetScipyEvaluationProvider",
    "TargetScipyHostEvaluation",
    "TargetScipyHostFinalizer",
    "inspect_target_scipy_device_packet_layout",
    "target_scipy_device_identity",
    "make_target_scipy_evaluation_provider",
]


_DecisionT = TypeVar("_DecisionT")
_DevicePacketT = TypeVar("_DevicePacketT")
_HostPacketT = TypeVar("_HostPacketT")
_DeviceValueT = TypeVar("_DeviceValueT")
_DeviceGradientT = TypeVar("_DeviceGradientT")
_HostValueT = TypeVar("_HostValueT")
_HostGradientT = TypeVar("_HostGradientT")


@dataclass(frozen=True)
class TargetScipyHostEvaluation(Generic[_HostValueT, _HostGradientT]):
    """Final host value and gradient returned to the SciPy adapter."""

    value: _HostValueT
    gradient: _HostGradientT
    lifecycle: TargetScipyEvaluationLifecycle | None = None


@dataclass(frozen=True, slots=True)
class TargetScipyDeviceIdentity:
    """Stable identity of the one device backing a packed evaluation."""

    platform: str
    device_kind: str
    device_id: int
    process_index: int


@dataclass(frozen=True, slots=True)
class TargetScipyDevicePacketLayout:
    """Concrete pre-transfer array and addressable-shard inventory."""

    array_leaf_count: int
    addressable_shard_count: int
    device: TargetScipyDeviceIdentity


def target_scipy_device_identity(device: jax.Device) -> TargetScipyDeviceIdentity:
    return TargetScipyDeviceIdentity(
        platform=device.platform,
        device_kind=device.device_kind,
        device_id=device.id,
        process_index=device.process_index,
    )


def inspect_target_scipy_device_packet_layout(
    device_packet: object,
) -> TargetScipyDevicePacketLayout:
    """Require a concrete, fully addressable, one-device array packet."""

    leaves = _target_scipy_device_packet_array_leaves(device_packet)
    jax_leaves = tuple(jax.tree.leaves(device_packet))
    if any(not isinstance(leaf, jax.Array) for leaf in jax_leaves) or sorted(
        map(id, jax_leaves)
    ) != sorted(map(id, leaves)):
        raise TypeError(
            "Target SciPy device packet containers must expose exactly the "
            "validated JAX array leaves to JAX tree traversal."
        )
    shards = tuple(shard for leaf in leaves for shard in leaf.addressable_shards)
    if any(
        not leaf.is_fully_addressable
        or len(leaf.addressable_shards) != 1
        or len(leaf.devices()) != 1
        for leaf in leaves
    ):
        raise ValueError(
            "Target SciPy device packet arrays must be fully addressable and "
            "single-device."
        )
    device_identities = {target_scipy_device_identity(shard.device) for shard in shards}
    if len(shards) != len(leaves) or len(device_identities) != 1:
        raise ValueError(
            "Target SciPy device packet addressable shards must share one device."
        )
    return TargetScipyDevicePacketLayout(
        array_leaf_count=len(leaves),
        addressable_shard_count=len(shards),
        device=next(iter(device_identities)),
    )


def _target_scipy_device_packet_array_leaves(packet: object) -> tuple[jax.Array, ...]:
    if isinstance(packet, jax.Array):
        return (packet,)
    if type(packet) is dict:
        if not packet or any(type(key) is not str for key in packet):
            raise TypeError(
                "Target SciPy device packet dictionaries must be non-empty and "
                "use string keys."
            )
        return tuple(
            leaf
            for value in packet.values()
            for leaf in _target_scipy_device_packet_array_leaves(value)
        )
    if type(packet) in (tuple, list):
        if not packet:
            raise TypeError("Target SciPy device packet containers must be non-empty.")
        return tuple(
            leaf
            for value in packet
            for leaf in _target_scipy_device_packet_array_leaves(value)
        )
    raise TypeError(
        "Target SciPy device packets must contain only JAX arrays in non-empty "
        "tuple, list, or string-keyed dictionary containers."
    )


def _require_host_numeric_leaf(value: object) -> None:
    if isinstance(value, (jax.Array, jax.core.Tracer)):
        raise TypeError(
            "Target SciPy host finalizers must return only host-resident leaves."
        )
    if type(value) is np.ndarray:
        if value.dtype.kind not in "iuf":
            raise TypeError(
                "Target SciPy host finalizers must return real numeric arrays."
            )
        return
    if isinstance(value, np.generic):
        scalar_dtype = np.dtype(type(value))
        if scalar_dtype.type is not type(value) or scalar_dtype.kind not in "iuf":
            raise TypeError(
                "Target SciPy host finalizers must return real numeric scalars."
            )
        return
    if type(value) in (int, float):
        return
    raise TypeError(
        "Target SciPy host finalizers must return numeric Python or NumPy leaves "
        "in approved containers."
    )


def _require_host_numeric_tree(value: object) -> None:
    if type(value) is dict:
        leaves = tuple(
            leaf for item in value.values() for leaf in _host_numeric_tree_leaves(item)
        )
    else:
        leaves = _host_numeric_tree_leaves(value)
    if not leaves:
        raise TypeError("Target SciPy host finalizers must return non-empty values.")
    for leaf in leaves:
        _require_host_numeric_leaf(leaf)


def _host_numeric_tree_leaves(value: object) -> tuple[object, ...]:
    if (
        type(value) is np.ndarray
        or isinstance(value, np.generic)
        or type(value) in (bool, int, float, complex)
        or isinstance(value, (jax.Array, jax.core.Tracer))
    ):
        return (value,)
    if type(value) is dict:
        return tuple(
            leaf for item in value.values() for leaf in _host_numeric_tree_leaves(item)
        )
    if type(value) in (tuple, list):
        return tuple(leaf for item in value for leaf in _host_numeric_tree_leaves(item))
    return (value,)


def _require_host_resident_target_scipy_evaluation(
    evaluation: TargetScipyHostEvaluation[_HostValueT, _HostGradientT],
) -> TargetScipyHostEvaluation[_HostValueT, _HostGradientT]:
    _require_host_numeric_tree(evaluation.value)
    _require_host_numeric_tree(evaluation.gradient)
    return evaluation


class TargetScipyHostFinalizer(Protocol[_HostPacketT, _HostValueT, _HostGradientT]):
    """Resolve one materialized packet using the original SciPy decision vector."""

    def __call__(
        self,
        host_decision_vector: np.ndarray,
        host_packet: _HostPacketT,
        device_layout: TargetScipyDevicePacketLayout,
        /,
    ) -> TargetScipyHostEvaluation[_HostValueT, _HostGradientT]: ...


@dataclass(frozen=True)
class TargetScipyDeviceEvaluation(
    Generic[_DevicePacketT, _HostPacketT, _HostValueT, _HostGradientT]
):
    """One complete device packet and its typed host finalizer."""

    device_packet: _DevicePacketT
    finalize_host: TargetScipyHostFinalizer[
        _HostPacketT,
        _HostValueT,
        _HostGradientT,
    ]


@runtime_checkable
class TargetScipyEvaluationProvider(
    Protocol[
        _DecisionT,
        _DevicePacketT,
        _HostPacketT,
        _DeviceValueT,
        _DeviceGradientT,
        _HostValueT,
        _HostGradientT,
    ]
):
    """Preserve tuple calls while exposing packed SciPy-target evaluation."""

    def __call__(
        self,
        decision_vector: _DecisionT,
        /,
    ) -> tuple[_DeviceValueT, _DeviceGradientT]: ...

    def evaluate_target_scipy(
        self,
        decision_vector: _DecisionT,
        host_decision_vector: np.ndarray,
        /,
    ) -> TargetScipyDeviceEvaluation[
        _DevicePacketT,
        _HostPacketT,
        _HostValueT,
        _HostGradientT,
    ]: ...


@dataclass(frozen=True)
class _FunctionalTargetScipyEvaluationProvider(
    Generic[
        _DecisionT,
        _DevicePacketT,
        _HostPacketT,
        _DeviceValueT,
        _DeviceGradientT,
        _HostValueT,
        _HostGradientT,
    ]
):
    value_and_grad: Callable[[_DecisionT], tuple[_DeviceValueT, _DeviceGradientT]]
    evaluate_device: Callable[[_DecisionT, np.ndarray], _DevicePacketT]
    finalize_host: TargetScipyHostFinalizer[
        _HostPacketT,
        _HostValueT,
        _HostGradientT,
    ]

    def __call__(
        self,
        decision_vector: _DecisionT,
        /,
    ) -> tuple[_DeviceValueT, _DeviceGradientT]:
        return self.value_and_grad(decision_vector)

    def evaluate_target_scipy(
        self,
        decision_vector: _DecisionT,
        host_decision_vector: np.ndarray,
        /,
    ) -> TargetScipyDeviceEvaluation[
        _DevicePacketT,
        _HostPacketT,
        _HostValueT,
        _HostGradientT,
    ]:
        return TargetScipyDeviceEvaluation(
            device_packet=self.evaluate_device(
                decision_vector,
                host_decision_vector,
            ),
            finalize_host=self.finalize_host,
        )


def make_target_scipy_evaluation_provider(
    value_and_grad: Callable[[_DecisionT], tuple[_DeviceValueT, _DeviceGradientT]],
    evaluate_device: Callable[[_DecisionT, np.ndarray], _DevicePacketT],
    finalize_host: TargetScipyHostFinalizer[
        _HostPacketT,
        _HostValueT,
        _HostGradientT,
    ],
) -> TargetScipyEvaluationProvider[
    _DecisionT,
    _DevicePacketT,
    _HostPacketT,
    _DeviceValueT,
    _DeviceGradientT,
    _HostValueT,
    _HostGradientT,
]:
    """Compose a compatible tuple callable and packed SciPy evaluator."""
    return _FunctionalTargetScipyEvaluationProvider(
        value_and_grad=value_and_grad,
        evaluate_device=evaluate_device,
        finalize_host=finalize_host,
    )
