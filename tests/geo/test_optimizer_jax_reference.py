from collections.abc import Callable
from typing import TypeAlias, TypedDict

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import OptimizeResult

from simsopt_jax.backend import target_lane_purity_active
import simsopt_jax.geo.optimizers._evaluation_provider as _provider
import simsopt_jax.geo.optimizers._shared as _opt_shared
import simsopt_jax.geo.optimizers.optimizer as _opt
import simsopt_jax.geo.optimizers.reference as _opt_ref


_HostVector: TypeAlias = NDArray[np.float64]
_ScipyObjective: TypeAlias = Callable[[_HostVector], tuple[np.float64, _HostVector]]
_ScipyCallback: TypeAlias = Callable[[_HostVector], None]


class _FlatDeviceEvaluationPacket(TypedDict):
    value: jax.Array
    gradient: jax.Array
    control: jax.Array


class _StructuredDeviceEvaluationPacket(TypedDict):
    value: jax.Array
    gradient: dict[str, jax.Array]
    control: jax.Array


class _FlatHostEvaluationPacket(TypedDict):
    value: np.ndarray
    gradient: np.ndarray
    control: np.ndarray


class _StructuredHostEvaluationPacket(TypedDict):
    value: np.ndarray
    gradient: dict[str, np.ndarray]
    control: np.ndarray


class _RecordingTargetScipyLifecycle:
    def __init__(self) -> None:
        self.classifications: list[_provider.TargetScipyEvaluationClass] = []
        self.outcomes: list[_provider.TargetScipyEvaluationOutcome] = []

    def classify_target_scipy_evaluation(
        self,
        evaluation_class: _provider.TargetScipyEvaluationClass,
        /,
    ) -> None:
        assert jax.config.jax_transfer_guard_host_to_device == "disallow"
        assert jax.config.jax_transfer_guard_device_to_host == "disallow"
        self.classifications.append(evaluation_class)

    def resolve_target_scipy_evaluation(
        self,
        outcome: _provider.TargetScipyEvaluationOutcome,
        /,
    ) -> None:
        assert jax.config.jax_transfer_guard_host_to_device == "disallow"
        assert jax.config.jax_transfer_guard_device_to_host == "disallow"
        self.outcomes.append(outcome)


def _make_recording_target_scipy_provider(
    lifecycles: list[_RecordingTargetScipyLifecycle],
):
    def value_and_grad(x: jax.Array) -> tuple[jax.Array, jax.Array]:
        return jnp.dot(x, x), 2.0 * x

    def evaluate_device(
        x: jax.Array,
        _host_decision_vector: np.ndarray,
    ) -> _FlatDeviceEvaluationPacket:
        return _FlatDeviceEvaluationPacket(
            value=jnp.dot(x, x),
            gradient=2.0 * x,
            control=jnp.asarray([1], dtype=jnp.int32),
        )

    def finalize_host(
        _host_decision_vector: np.ndarray,
        packet: _FlatHostEvaluationPacket,
        _device_layout: _provider.TargetScipyDevicePacketLayout,
    ) -> _provider.TargetScipyHostEvaluation[np.ndarray, np.ndarray]:
        assert jax.config.jax_transfer_guard_host_to_device == "disallow"
        assert jax.config.jax_transfer_guard_device_to_host == "disallow"
        lifecycle = _RecordingTargetScipyLifecycle()
        lifecycles.append(lifecycle)
        return _provider.TargetScipyHostEvaluation(
            value=packet["value"],
            gradient=packet["gradient"],
            lifecycle=lifecycle,
        )

    return _provider.make_target_scipy_evaluation_provider(
        value_and_grad,
        evaluate_device,
        finalize_host,
    )


def test_target_scipy_tuple_callable_materializes_value_and_gradient_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materialized_packets: list[object] = []
    device_get = jax.device_get

    def record_device_get(packet: object) -> object:
        materialized_packets.append(packet)
        return device_get(packet)

    monkeypatch.setattr(_opt_ref.jax, "device_get", record_device_get)

    def value_and_grad(x: jax.Array) -> tuple[jax.Array, jax.Array]:
        return jnp.dot(x, x), 2.0 * x

    scipy_fun = _opt_ref._make_scipy_host_value_and_grad_objective(
        value_and_grad,
        x_dtype=np.dtype(np.float64),
        initial_call={},
    )
    value, gradient = scipy_fun(np.asarray([2.0, -1.0], dtype=np.float64))

    assert len(materialized_packets) == 1
    assert isinstance(materialized_packets[0], tuple)
    assert value == np.float64(5.0)
    np.testing.assert_array_equal(
        gradient,
        np.asarray([4.0, -2.0], dtype=np.float64),
    )


def test_target_scipy_provider_materializes_complete_packet_before_finalizing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    materialized_packets: list[object] = []
    host_decision_vector = np.asarray([2.0, -1.0], dtype=np.float64)
    device_get = jax.device_get

    def record_device_get(packet: object) -> object:
        events.append("device_get")
        materialized_packets.append(packet)
        return device_get(packet)

    monkeypatch.setattr(_opt_ref.jax, "device_get", record_device_get)

    def value_and_grad(x: jax.Array) -> tuple[jax.Array, jax.Array]:
        return jnp.dot(x, x), 2.0 * x

    def evaluate_device(
        x: jax.Array,
        scipy_decision_vector: np.ndarray,
    ) -> _FlatDeviceEvaluationPacket:
        events.append("evaluate_target_scipy")
        assert scipy_decision_vector is not host_decision_vector
        assert not scipy_decision_vector.flags.writeable
        np.testing.assert_array_equal(scipy_decision_vector, host_decision_vector)
        return _FlatDeviceEvaluationPacket(
            value=jnp.dot(x, x),
            gradient=2.0 * x,
            control=jnp.asarray([7, 1], dtype=jnp.int32),
        )

    def finalize_host(
        scipy_decision_vector: np.ndarray,
        packet: _FlatHostEvaluationPacket,
        device_layout: _provider.TargetScipyDevicePacketLayout,
    ) -> _provider.TargetScipyHostEvaluation[np.ndarray, np.ndarray]:
        events.append("finalize_host")
        assert scipy_decision_vector is not host_decision_vector
        assert not scipy_decision_vector.flags.writeable
        np.testing.assert_array_equal(scipy_decision_vector, host_decision_vector)
        assert isinstance(packet["value"], np.ndarray)
        assert isinstance(packet["gradient"], np.ndarray)
        assert isinstance(packet["control"], np.ndarray)
        assert device_layout.array_leaf_count == 3
        assert device_layout.addressable_shard_count == 3
        assert device_layout.device.platform == jax.default_backend()
        np.testing.assert_array_equal(packet["control"], np.asarray([7, 1]))
        return _provider.TargetScipyHostEvaluation(
            value=packet["value"],
            gradient=packet["gradient"],
        )

    provider = _provider.make_target_scipy_evaluation_provider(
        value_and_grad,
        evaluate_device,
        finalize_host,
    )
    direct_value, direct_gradient = provider(
        jnp.asarray([2.0, -1.0], dtype=jnp.float64)
    )
    assert np.asarray(direct_value) == np.float64(5.0)
    np.testing.assert_array_equal(
        np.asarray(direct_gradient),
        np.asarray([4.0, -2.0], dtype=np.float64),
    )
    scipy_fun = _opt_ref._make_scipy_host_value_and_grad_objective(
        provider,
        x_dtype=np.dtype(np.float64),
        initial_call={},
    )
    value, gradient = scipy_fun(host_decision_vector)

    assert events == ["evaluate_target_scipy", "device_get", "finalize_host"]
    assert len(materialized_packets) == 1
    assert type(materialized_packets[0]) is dict
    assert set(materialized_packets[0]) == {"value", "gradient", "control"}
    assert value == np.float64(5.0)
    np.testing.assert_array_equal(
        gradient,
        np.asarray([4.0, -2.0], dtype=np.float64),
    )


def test_invalid_target_scipy_packet_rejects_before_transfer_and_finalizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device_get_count = 0
    finalizer_count = 0

    def unexpected_device_get(_packet: object) -> object:
        nonlocal device_get_count
        device_get_count += 1
        raise AssertionError("invalid packets must reject before device_get")

    def unexpected_finalizer(
        _host_decision_vector: np.ndarray,
        _host_packet: object,
        _device_layout: _provider.TargetScipyDevicePacketLayout,
    ) -> _provider.TargetScipyHostEvaluation[np.ndarray, np.ndarray]:
        nonlocal finalizer_count
        finalizer_count += 1
        raise AssertionError("invalid packets must reject before finalization")

    monkeypatch.setattr(_opt_ref.jax, "device_get", unexpected_device_get)
    evaluation = _provider.TargetScipyDeviceEvaluation(
        device_packet=(np.asarray([1.0]),),
        finalize_host=unexpected_finalizer,
    )

    with pytest.raises(TypeError, match="only JAX arrays"):
        _opt_ref._materialize_target_scipy_evaluation(
            evaluation,
            host_decision_vector=np.asarray([1.0]),
            expected_device=_provider.target_scipy_device_identity(jax.devices()[0]),
        )

    assert device_get_count == 0
    assert finalizer_count == 0


def test_target_scipy_device_packet_layout_rejects_host_leaf() -> None:
    with pytest.raises(TypeError, match="only JAX arrays"):
        _provider.inspect_target_scipy_device_packet_layout((np.asarray([1.0]),))


@pytest.mark.parametrize("hidden_payload", [None, (), []])
def test_target_scipy_device_packet_layout_rejects_hidden_host_payload(
    hidden_payload: object,
) -> None:
    with pytest.raises(TypeError, match="non-empty|only JAX arrays"):
        _provider.inspect_target_scipy_device_packet_layout(
            (jnp.asarray([1.0]), hidden_payload)
        )


def test_target_scipy_device_packet_layout_rejects_opaque_container_subclass() -> None:
    class PacketList(list[jax.Array]):
        pass

    with pytest.raises(TypeError, match="containers|JAX tree traversal"):
        _provider.inspect_target_scipy_device_packet_layout(
            PacketList([jnp.asarray([1.0])])
        )


def test_target_scipy_device_packet_rejects_string_key_subclass() -> None:
    comparison_attempted = False

    class AdversarialString(str):
        def __lt__(self, other: object) -> bool:
            nonlocal comparison_attempted
            comparison_attempted = True
            return str.__lt__(self, other)

    packet = {
        AdversarialString("value"): jnp.asarray([1.0]),
        AdversarialString("gradient"): jnp.asarray([2.0]),
    }

    with pytest.raises(TypeError, match="string keys"):
        _provider.inspect_target_scipy_device_packet_layout(packet)

    assert comparison_attempted is False


def test_target_scipy_device_packet_rejects_adversarial_named_tuple() -> None:
    iteration_attempted = False

    class AdversarialNamedTuple(tuple[jax.Array]):
        _fields = ("value",)

        def __new__(cls, value: jax.Array):
            return tuple.__new__(cls, (value,))

        def __iter__(self):
            nonlocal iteration_attempted
            iteration_attempted = True
            return tuple.__iter__(self)

    with pytest.raises(TypeError, match="only JAX arrays"):
        _provider.inspect_target_scipy_device_packet_layout(
            AdversarialNamedTuple(jnp.asarray([1.0]))
        )

    assert iteration_attempted is False


def test_target_scipy_host_evaluation_rejects_device_value_in_opaque_container() -> (
    None
):
    class HostList(list[object]):
        pass

    evaluation = _provider.TargetScipyHostEvaluation(
        value=np.float64(1.0),
        gradient=HostList([jnp.asarray([1.0])]),
    )

    with pytest.raises(TypeError, match="numeric Python or NumPy"):
        _provider._require_host_resident_target_scipy_evaluation(evaluation)


def test_target_scipy_host_evaluation_rejects_opaque_array_wrapper() -> None:
    class OpaqueHostArray:
        def __init__(self, array: jax.Array) -> None:
            self.array = array

        def __array__(self) -> np.ndarray:
            return np.asarray(self.array)

    evaluation = _provider.TargetScipyHostEvaluation(
        value=np.float64(1.0),
        gradient=OpaqueHostArray(jnp.asarray([1.0])),
    )

    with pytest.raises(TypeError, match="numeric Python or NumPy"):
        _provider._require_host_resident_target_scipy_evaluation(evaluation)


def test_target_scipy_host_evaluation_rejects_adversarial_named_tuple() -> None:
    class AdversarialNamedTuple(tuple[object]):
        _fields = ("gradient",)

        def __new__(cls, gradient: jax.Array):
            return tuple.__new__(cls, (gradient,))

        def __iter__(self):
            return iter((np.asarray([1.0]),))

        def __array__(self) -> np.ndarray:
            return np.asarray(tuple.__getitem__(self, 0))

    evaluation = _provider.TargetScipyHostEvaluation(
        value=np.float64(1.0),
        gradient=AdversarialNamedTuple(jnp.asarray([1.0])),
    )

    with pytest.raises(TypeError, match="numeric Python or NumPy"):
        _provider._require_host_resident_target_scipy_evaluation(evaluation)


@pytest.mark.parametrize(
    "invalid_gradient",
    [
        np.asarray([1.0 + 2.0j]),
        np.asarray(["1.5"]),
        np.asarray([np.datetime64("2020-01-01")]),
        np.asarray([True]),
        np.str_("1.5"),
        np.datetime64("2020-01-01"),
    ],
)
def test_public_target_scipy_provider_rejects_nonreal_host_gradient(
    invalid_gradient: object,
) -> None:
    def value_and_grad(x: jax.Array) -> tuple[jax.Array, jax.Array]:
        return jnp.dot(x, x), 2.0 * x

    def evaluate_device(
        x: jax.Array,
        _host_decision_vector: np.ndarray,
    ) -> _FlatDeviceEvaluationPacket:
        return _FlatDeviceEvaluationPacket(
            value=jnp.dot(x, x),
            gradient=2.0 * x,
            control=jnp.asarray([1], dtype=jnp.int32),
        )

    def finalize_host(
        _host_decision_vector: np.ndarray,
        packet: _FlatHostEvaluationPacket,
        _device_layout: _provider.TargetScipyDevicePacketLayout,
    ) -> _provider.TargetScipyHostEvaluation[np.ndarray, object]:
        return _provider.TargetScipyHostEvaluation(
            value=packet["value"],
            gradient=invalid_gradient,
        )

    provider = _provider.make_target_scipy_evaluation_provider(
        value_and_grad,
        evaluate_device,
        finalize_host,
    )
    scipy_fun = _opt_ref._make_scipy_host_value_and_grad_objective(
        provider,
        x_dtype=np.dtype(np.float64),
        initial_call={},
    )

    with pytest.raises(TypeError, match="real numeric"):
        scipy_fun(np.asarray([2.0, -1.0], dtype=np.float64))


def test_target_scipy_host_evaluation_rejects_numpy_scalar_subclass() -> None:
    dtype_accessed = False

    class AdversarialFloat(np.float64):
        @property
        def dtype(self) -> np.dtype:
            nonlocal dtype_accessed
            dtype_accessed = True
            return np.dtype(np.float64)

    evaluation = _provider.TargetScipyHostEvaluation(
        value=np.float64(1.0),
        gradient=AdversarialFloat(2.0),
    )

    with pytest.raises(TypeError, match="real numeric scalars"):
        _provider._require_host_resident_target_scipy_evaluation(evaluation)

    assert dtype_accessed is False


def test_target_scipy_device_packet_rejects_nondecision_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = jax.devices()[0]
    actual_identity = _provider.target_scipy_device_identity(device)
    wrong_identity = _provider.TargetScipyDeviceIdentity(
        platform=actual_identity.platform,
        device_kind=actual_identity.device_kind,
        device_id=actual_identity.device_id + 1,
        process_index=actual_identity.process_index,
    )
    finalizer_called = False

    def forbidden_finalizer(
        _host_decision_vector: np.ndarray,
        _host_packet: object,
        _device_layout: _provider.TargetScipyDevicePacketLayout,
    ) -> _provider.TargetScipyHostEvaluation[np.ndarray, np.ndarray]:
        nonlocal finalizer_called
        finalizer_called = True
        raise AssertionError("wrong-device packets must reject before finalization")

    evaluation = _provider.TargetScipyDeviceEvaluation(
        device_packet=(jnp.asarray([1.0]),),
        finalize_host=forbidden_finalizer,
    )
    device_get = jax.device_get
    device_get_count = 0

    def record_device_get(value: object) -> object:
        nonlocal device_get_count
        device_get_count += 1
        return device_get(value)

    monkeypatch.setattr(_opt_ref.jax, "device_get", record_device_get)

    with pytest.raises(ValueError, match="share the decision vector device"):
        _opt_ref._materialize_target_scipy_evaluation(
            evaluation,
            host_decision_vector=np.asarray([1.0]),
            expected_device=wrong_identity,
        )

    assert device_get_count == 0
    assert finalizer_called is False


def test_target_scipy_provider_mutation_cannot_change_authoritative_snapshot() -> None:
    scipy_decision_vector = np.asarray([2.0, -1.0], dtype=np.float64)
    extension_snapshots: list[np.ndarray] = []

    def value_and_grad(x: jax.Array) -> tuple[jax.Array, jax.Array]:
        return jnp.dot(x, x), 2.0 * x

    def mutate_isolated_device_input(
        x: jax.Array,
        host_decision_snapshot: np.ndarray,
    ) -> _FlatDeviceEvaluationPacket:
        extension_snapshots.append(host_decision_snapshot)
        host_decision_snapshot.flags.writeable = True
        host_decision_snapshot[0] = 99.0
        return _FlatDeviceEvaluationPacket(
            value=jnp.dot(x, x),
            gradient=2.0 * x,
            control=jnp.asarray([1], dtype=jnp.int32),
        )

    def finalize_host(
        host_decision_snapshot: np.ndarray,
        packet: _FlatHostEvaluationPacket,
        _device_layout: _provider.TargetScipyDevicePacketLayout,
    ) -> _provider.TargetScipyHostEvaluation[np.ndarray, np.ndarray]:
        np.testing.assert_array_equal(host_decision_snapshot, scipy_decision_vector)
        extension_snapshots.append(host_decision_snapshot)
        host_decision_snapshot.flags.writeable = True
        host_decision_snapshot[0] = 77.0
        return _provider.TargetScipyHostEvaluation(
            value=packet["value"],
            gradient=packet["gradient"],
        )

    provider = _provider.make_target_scipy_evaluation_provider(
        value_and_grad,
        mutate_isolated_device_input,
        finalize_host,
    )
    scipy_fun = _opt_ref._make_scipy_host_value_and_grad_objective(
        provider,
        x_dtype=np.dtype(np.float64),
        initial_call={},
    )

    value, gradient = scipy_fun(scipy_decision_vector)

    assert value == np.float64(5.0)
    np.testing.assert_array_equal(gradient, np.asarray([4.0, -2.0]))
    np.testing.assert_array_equal(
        scipy_decision_vector,
        np.asarray([2.0, -1.0], dtype=np.float64),
    )
    assert len(extension_snapshots) == 2
    assert extension_snapshots[0] is not extension_snapshots[1]


def test_target_scipy_device_packet_layout_rejects_multidevice_shards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDevice:
        def __init__(self, device_id: int) -> None:
            self.platform = "gpu"
            self.device_kind = "fake-gpu"
            self.id = device_id
            self.process_index = 0

    class FakeShard:
        def __init__(self, device: FakeDevice) -> None:
            self.device = device

    class FakeArray:
        def __init__(self) -> None:
            self.is_fully_addressable = True
            self.addressable_shards = (
                FakeShard(FakeDevice(0)),
                FakeShard(FakeDevice(1)),
            )

        def devices(self) -> set[FakeDevice]:
            return {shard.device for shard in self.addressable_shards}

    monkeypatch.setattr(_provider.jax, "Array", FakeArray)

    with pytest.raises(ValueError, match="single-device"):
        _provider.inspect_target_scipy_device_packet_layout((FakeArray(),))


def test_target_scipy_device_packet_layout_rejects_nonaddressable_array(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDevice:
        platform = "gpu"
        device_kind = "fake-gpu"
        id = 0
        process_index = 0

    class FakeShard:
        device = FakeDevice()

    class FakeArray:
        is_fully_addressable = False
        addressable_shards = (FakeShard(),)

        def devices(self) -> set[FakeDevice]:
            return {self.addressable_shards[0].device}

    monkeypatch.setattr(_provider.jax, "Array", FakeArray)

    with pytest.raises(ValueError, match="fully addressable"):
        _provider.inspect_target_scipy_device_packet_layout((FakeArray(),))


def test_target_scipy_device_packet_layout_rejects_cross_device_packet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDevice:
        def __init__(self, device_id: int) -> None:
            self.platform = "gpu"
            self.device_kind = "fake-gpu"
            self.id = device_id
            self.process_index = 0

    class FakeShard:
        def __init__(self, device: FakeDevice) -> None:
            self.device = device

    class FakeArray:
        def __init__(self, device_id: int) -> None:
            self.is_fully_addressable = True
            self.addressable_shards = (FakeShard(FakeDevice(device_id)),)

        def devices(self) -> set[FakeDevice]:
            return {self.addressable_shards[0].device}

    monkeypatch.setattr(_provider.jax, "Array", FakeArray)

    with pytest.raises(ValueError, match="must share one device"):
        _provider.inspect_target_scipy_device_packet_layout(
            (FakeArray(0), FakeArray(1))
        )


def test_target_scipy_provider_lifecycle_tracks_only_scipy_trials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycles: list[_RecordingTargetScipyLifecycle] = []

    def value_and_grad(x: jax.Array) -> tuple[jax.Array, jax.Array]:
        return jnp.dot(x, x), 2.0 * x

    def evaluate_device(
        x: jax.Array,
        _host_decision_vector: np.ndarray,
    ) -> _FlatDeviceEvaluationPacket:
        return _FlatDeviceEvaluationPacket(
            value=jnp.dot(x, x),
            gradient=2.0 * x,
            control=jnp.asarray([1], dtype=jnp.int32),
        )

    def finalize_host(
        _host_decision_vector: np.ndarray,
        packet: _FlatHostEvaluationPacket,
        _device_layout: _provider.TargetScipyDevicePacketLayout,
    ) -> _provider.TargetScipyHostEvaluation[np.ndarray, np.ndarray]:
        lifecycle = _RecordingTargetScipyLifecycle()
        lifecycles.append(lifecycle)
        return _provider.TargetScipyHostEvaluation(
            value=packet["value"],
            gradient=packet["gradient"],
            lifecycle=lifecycle,
        )

    provider = _provider.make_target_scipy_evaluation_provider(
        value_and_grad,
        evaluate_device,
        finalize_host,
    )

    def fake_scipy_minimize(
        fun: _ScipyObjective,
        x0: _HostVector,
        *,
        jac: bool,
        method: str,
        options: dict[str, float | int],
        callback: _ScipyCallback | None,
    ) -> OptimizeResult:
        del method, options
        assert jac is True
        assert callback is not None
        fun(x0)
        fun(x0 + np.asarray([0.25, -0.25], dtype=np.float64))
        accepted = x0 + np.asarray([0.5, -0.5], dtype=np.float64)
        accepted_fun, accepted_gradient = fun(accepted)
        fun(x0 + np.asarray([0.75, -0.75], dtype=np.float64))
        callback(accepted)
        fun(x0 + np.asarray([1.0, -1.0], dtype=np.float64))
        return OptimizeResult(
            x=accepted,
            fun=accepted_fun,
            jac=accepted_gradient,
            nit=1,
            nfev=5,
            njev=5,
            success=True,
            status=0,
            message="test convergence",
        )

    monkeypatch.setattr(_opt_ref, "scipy_minimize", fake_scipy_minimize)
    result = _opt_ref.target_scipy_minimize_value_and_grad(
        provider,
        np.asarray([2.0, -1.0], dtype=np.float64),
        method="lbfgs",
        tol=1.0e-10,
        maxiter=2,
        options={},
    )

    assert result.scipy_call_contract["callback"] is None
    assert [lifecycle.classifications for lifecycle in lifecycles] == [
        [_provider.TargetScipyEvaluationClass.INITIAL_OPTIMIZER_EVALUATION],
        [_provider.TargetScipyEvaluationClass.OPTIMIZER_TRIAL],
        [_provider.TargetScipyEvaluationClass.OPTIMIZER_TRIAL],
        [_provider.TargetScipyEvaluationClass.OPTIMIZER_TRIAL],
        [_provider.TargetScipyEvaluationClass.OPTIMIZER_TRIAL],
    ]
    assert [lifecycle.outcomes for lifecycle in lifecycles] == [
        [],
        [_provider.TargetScipyEvaluationOutcome.REJECTED],
        [_provider.TargetScipyEvaluationOutcome.ACCEPTED],
        [_provider.TargetScipyEvaluationOutcome.REJECTED],
        [_provider.TargetScipyEvaluationOutcome.REJECTED],
    ]


def test_target_scipy_lifecycle_completion_rejects_all_pending_trials_once() -> None:
    controller = _opt_ref._TargetScipyEvaluationLifecycleController()
    initial = _RecordingTargetScipyLifecycle()
    first_trial = _RecordingTargetScipyLifecycle()
    second_trial = _RecordingTargetScipyLifecycle()
    x0 = np.asarray([2.0, -1.0], dtype=np.float64)

    controller.evaluation_returned(x0, initial)
    controller.evaluation_returned(x0 + 0.25, first_trial)
    controller.evaluation_returned(x0 + 0.5, second_trial)
    controller.optimizer_completed()
    controller.optimizer_completed()

    assert initial.outcomes == []
    assert first_trial.outcomes == [_provider.TargetScipyEvaluationOutcome.REJECTED]
    assert second_trial.outcomes == [_provider.TargetScipyEvaluationOutcome.REJECTED]


def test_target_scipy_lifecycle_accepts_latest_exact_duplicate() -> None:
    controller = _opt_ref._TargetScipyEvaluationLifecycleController()
    initial = _RecordingTargetScipyLifecycle()
    first_duplicate = _RecordingTargetScipyLifecycle()
    different_trial = _RecordingTargetScipyLifecycle()
    latest_duplicate = _RecordingTargetScipyLifecycle()
    x0 = np.asarray([2.0, -1.0], dtype=np.float64)
    duplicate_x = np.asarray([1.0, -0.5], dtype=np.float64)

    controller.evaluation_returned(x0, initial)
    controller.evaluation_returned(duplicate_x, first_duplicate)
    controller.evaluation_returned(x0 + 0.25, different_trial)
    controller.evaluation_returned(duplicate_x, latest_duplicate)
    controller.accepted_trial(duplicate_x)

    assert initial.outcomes == []
    assert first_duplicate.outcomes == [_provider.TargetScipyEvaluationOutcome.REJECTED]
    assert different_trial.outcomes == [_provider.TargetScipyEvaluationOutcome.REJECTED]
    assert latest_duplicate.outcomes == [
        _provider.TargetScipyEvaluationOutcome.ACCEPTED
    ]


def test_public_target_scipy_provider_distinguishes_signed_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycles: list[_RecordingTargetScipyLifecycle] = []
    provider = _make_recording_target_scipy_provider(lifecycles)

    def fake_scipy_minimize(
        fun: _ScipyObjective,
        x0: _HostVector,
        *,
        jac: bool,
        method: str,
        options: dict[str, float | int],
        callback: _ScipyCallback | None,
    ) -> OptimizeResult:
        del method, options
        assert jac is True
        assert callback is not None
        fun(x0)
        positive_zero = np.asarray([0.0], dtype=np.float64)
        negative_zero = np.asarray([-0.0], dtype=np.float64)
        final_fun, final_gradient = fun(positive_zero)
        fun(negative_zero)
        callback(positive_zero)
        return OptimizeResult(
            x=positive_zero,
            fun=final_fun,
            jac=final_gradient,
            nit=1,
            nfev=3,
            njev=3,
            success=True,
            status=0,
            message="test convergence",
        )

    monkeypatch.setattr(_opt_ref, "scipy_minimize", fake_scipy_minimize)
    _opt_ref.target_scipy_minimize_value_and_grad(
        provider,
        np.asarray([1.0], dtype=np.float64),
        method="lbfgs",
        tol=1.0e-10,
        maxiter=1,
        options={},
    )

    assert [lifecycle.outcomes for lifecycle in lifecycles] == [
        [],
        [_provider.TargetScipyEvaluationOutcome.ACCEPTED],
        [_provider.TargetScipyEvaluationOutcome.REJECTED],
    ]


def test_public_target_scipy_provider_accepts_latest_byte_exact_duplicate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycles: list[_RecordingTargetScipyLifecycle] = []
    provider = _make_recording_target_scipy_provider(lifecycles)

    def fake_scipy_minimize(
        fun: _ScipyObjective,
        x0: _HostVector,
        *,
        jac: bool,
        method: str,
        options: dict[str, float | int],
        callback: _ScipyCallback | None,
    ) -> OptimizeResult:
        del method, options
        assert jac is True
        assert callback is not None
        fun(x0)
        duplicate = np.asarray([0.0], dtype=np.float64)
        fun(duplicate)
        final_fun, final_gradient = fun(duplicate.copy())
        callback(duplicate)
        return OptimizeResult(
            x=duplicate,
            fun=final_fun,
            jac=final_gradient,
            nit=1,
            nfev=3,
            njev=3,
            success=True,
            status=0,
            message="test convergence",
        )

    monkeypatch.setattr(_opt_ref, "scipy_minimize", fake_scipy_minimize)
    _opt_ref.target_scipy_minimize_value_and_grad(
        provider,
        np.asarray([1.0], dtype=np.float64),
        method="lbfgs",
        tol=1.0e-10,
        maxiter=1,
        options={},
    )

    assert [lifecycle.outcomes for lifecycle in lifecycles] == [
        [],
        [_provider.TargetScipyEvaluationOutcome.REJECTED],
        [_provider.TargetScipyEvaluationOutcome.ACCEPTED],
    ]


def test_public_target_scipy_provider_leaves_unknown_callback_unresolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycles: list[_RecordingTargetScipyLifecycle] = []
    provider = _make_recording_target_scipy_provider(lifecycles)

    def fake_scipy_minimize(
        fun: _ScipyObjective,
        x0: _HostVector,
        *,
        jac: bool,
        method: str,
        options: dict[str, float | int],
        callback: _ScipyCallback | None,
    ) -> OptimizeResult:
        del method, options
        assert jac is True
        assert callback is not None
        fun(x0)
        fun(x0 + 0.5)
        callback(x0 + 1.0)
        raise AssertionError("unknown callback must fail before SciPy returns")

    monkeypatch.setattr(_opt_ref, "scipy_minimize", fake_scipy_minimize)

    with pytest.raises(RuntimeError, match="no exact objective evaluation"):
        _opt_ref.target_scipy_minimize_value_and_grad(
            provider,
            np.asarray([2.0, -1.0], dtype=np.float64),
            method="lbfgs",
            tol=1.0e-10,
            maxiter=1,
            options={},
        )

    assert [lifecycle.classifications for lifecycle in lifecycles] == [
        [_provider.TargetScipyEvaluationClass.INITIAL_OPTIMIZER_EVALUATION],
        [_provider.TargetScipyEvaluationClass.OPTIMIZER_TRIAL],
    ]
    assert [lifecycle.outcomes for lifecycle in lifecycles] == [[], []]


def test_scipy_unknown_callback_leaves_provider_trial_unresolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _opt_ref._TargetScipyEvaluationLifecycleController()
    initial = _RecordingTargetScipyLifecycle()
    trial = _RecordingTargetScipyLifecycle()
    lifecycles = iter((initial, trial))

    def value_and_grad(x: ArrayLike) -> tuple[np.float64, _HostVector]:
        x_host = np.asarray(x, dtype=np.float64)
        controller.evaluation_returned(x_host, next(lifecycles))
        return np.float64(np.dot(x_host, x_host)), 2.0 * x_host

    def fake_scipy_minimize(
        fun: _ScipyObjective,
        x0: _HostVector,
        *,
        jac: bool,
        method: str,
        options: dict[str, float | int],
        callback: _ScipyCallback | None,
    ) -> OptimizeResult:
        del method, options
        assert jac is True
        assert callback is not None
        fun(x0)
        fun(x0 + 0.5)
        callback(x0 + 1.0)
        raise AssertionError("unknown callback must fail before SciPy returns")

    monkeypatch.setattr(_opt_ref, "scipy_minimize", fake_scipy_minimize)

    with pytest.raises(RuntimeError, match="no exact objective evaluation"):
        _opt_ref._scipy_dispatch_core(
            value_and_grad,
            np.asarray([2.0, -1.0], dtype=np.float64),
            method="lbfgs",
            tol=1.0e-10,
            maxiter=1,
            options={},
            lifecycle_controller=controller,
        )

    assert initial.classifications == [
        _provider.TargetScipyEvaluationClass.INITIAL_OPTIMIZER_EVALUATION
    ]
    assert trial.classifications == [
        _provider.TargetScipyEvaluationClass.OPTIMIZER_TRIAL
    ]
    assert initial.outcomes == []
    assert trial.outcomes == []


def test_target_scipy_provider_survives_strict_purity_and_pytree_adaptation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SIMSOPT_TARGET_LANE_STRICT", "1")
    events: list[str] = []
    device_get_calls = 0
    host_decision_vector = np.asarray([0.5, 2.0, -1.0], dtype=np.float64)
    device_get = jax.device_get

    def record_device_get(packet: object) -> object:
        nonlocal device_get_calls
        device_get_calls += 1
        assert type(packet) is dict
        assert set(packet) == {"value", "gradient", "control"}
        return device_get(packet)

    monkeypatch.setattr(_opt_ref.jax, "device_get", record_device_get)

    def value_and_grad(
        state: dict[str, jax.Array],
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        assert target_lane_purity_active() is True
        events.append("value_and_grad")
        surface = state["surface"]
        iota = state["iota"]
        return (
            jnp.dot(surface, surface) + jnp.square(iota),
            {
                "surface": 2.0 * surface,
                "iota": 2.0 * iota,
            },
        )

    def evaluate_device(
        state: dict[str, jax.Array],
        scipy_decision_vector: np.ndarray,
    ) -> _StructuredDeviceEvaluationPacket:
        assert target_lane_purity_active() is True
        events.append("evaluate_device")
        assert scipy_decision_vector is not host_decision_vector
        assert not scipy_decision_vector.flags.writeable
        np.testing.assert_array_equal(scipy_decision_vector, host_decision_vector)
        surface = state["surface"]
        iota = state["iota"]
        return _StructuredDeviceEvaluationPacket(
            value=jnp.dot(surface, surface) + jnp.square(iota),
            gradient={
                "surface": 2.0 * surface,
                "iota": 2.0 * iota,
            },
            control=jnp.asarray([11], dtype=jnp.int32),
        )

    def finalize_host(
        scipy_decision_vector: np.ndarray,
        packet: _StructuredHostEvaluationPacket,
        _device_layout: _provider.TargetScipyDevicePacketLayout,
    ) -> _provider.TargetScipyHostEvaluation[
        np.ndarray,
        dict[str, np.ndarray],
    ]:
        assert target_lane_purity_active() is False
        events.append("finalize_host")
        assert scipy_decision_vector is not host_decision_vector
        assert not scipy_decision_vector.flags.writeable
        np.testing.assert_array_equal(scipy_decision_vector, host_decision_vector)
        return _provider.TargetScipyHostEvaluation(
            value=packet["value"],
            gradient=packet["gradient"],
        )

    provider = _provider.make_target_scipy_evaluation_provider(
        value_and_grad,
        evaluate_device,
        finalize_host,
    )
    strict_provider = _opt.wrap_strict_target_lane_value_and_grad(provider)
    x0 = {
        "surface": jnp.asarray([2.0, -1.0], dtype=jnp.float64),
        "iota": jnp.asarray(0.5, dtype=jnp.float64),
    }
    flat_provider, flat_x0, _callback, adapter = (
        _opt_shared._prepare_optimizer_callable_inputs(
            strict_provider,
            x0,
            value_and_grad=True,
            callback=None,
        )
    )

    direct_value, direct_gradient = flat_provider(flat_x0)
    scipy_fun = _opt_ref._make_scipy_host_value_and_grad_objective(
        flat_provider,
        x_dtype=np.dtype(np.float64),
        initial_call={},
    )
    host_value, host_gradient = scipy_fun(host_decision_vector)
    expected_gradient, _ = ravel_pytree(
        {
            "surface": 2.0 * x0["surface"],
            "iota": 2.0 * x0["iota"],
        }
    )

    assert adapter is not None
    assert isinstance(flat_provider, _provider.TargetScipyEvaluationProvider)
    assert events == ["value_and_grad", "evaluate_device", "finalize_host"]
    assert device_get_calls == 1
    assert np.asarray(direct_value) == np.float64(5.25)
    np.testing.assert_array_equal(
        np.asarray(direct_gradient),
        np.asarray(expected_gradient),
    )
    assert host_value == np.float64(5.25)
    np.testing.assert_array_equal(
        host_gradient,
        np.asarray(expected_gradient),
    )


def test_target_scipy_provider_rejects_device_leaves_from_host_finalizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retained_device_evaluations: list[_FlatDeviceEvaluationPacket] = []
    device_get_calls = 0
    device_get = jax.device_get

    def record_device_get(packet: object) -> object:
        nonlocal device_get_calls
        device_get_calls += 1
        return device_get(packet)

    monkeypatch.setattr(_opt_ref.jax, "device_get", record_device_get)

    def value_and_grad(x: jax.Array) -> tuple[jax.Array, jax.Array]:
        return jnp.dot(x, x), 2.0 * x

    def evaluate_device(
        x: jax.Array,
        _host_decision_vector: np.ndarray,
    ) -> _FlatDeviceEvaluationPacket:
        packet = _FlatDeviceEvaluationPacket(
            value=jnp.dot(x, x),
            gradient=2.0 * x,
            control=jnp.asarray([1], dtype=jnp.int32),
        )
        retained_device_evaluations.append(packet)
        return packet

    def invalid_finalize_host(
        _host_decision_vector: np.ndarray,
        _host_packet: _FlatHostEvaluationPacket,
        _device_layout: _provider.TargetScipyDevicePacketLayout,
    ) -> _provider.TargetScipyHostEvaluation[jax.Array, jax.Array]:
        retained = retained_device_evaluations[-1]
        return _provider.TargetScipyHostEvaluation(
            value=retained["value"],
            gradient=retained["gradient"],
        )

    provider = _provider.make_target_scipy_evaluation_provider(
        value_and_grad,
        evaluate_device,
        invalid_finalize_host,
    )
    scipy_fun = _opt_ref._make_scipy_host_value_and_grad_objective(
        provider,
        x_dtype=np.dtype(np.float64),
        initial_call={},
    )

    with pytest.raises(TypeError, match="only host-resident leaves"):
        scipy_fun(np.asarray([2.0, -1.0], dtype=np.float64))

    assert device_get_calls == 1


@pytest.mark.parametrize(
    ("x", "fun", "jac"),
    (
        (np.asarray([np.nan, 1.0], dtype=np.float64), 3.4, np.asarray([0.0, 0.0])),
        (np.asarray([0.5, 1.0], dtype=np.float64), np.nan, np.asarray([0.0, 0.0])),
        (
            np.asarray([0.5, 1.0], dtype=np.float64),
            3.4,
            np.asarray([np.nan, 0.0], dtype=np.float64),
        ),
    ),
)
def test_normalize_scipy_result_marks_nonfinite_state_failure(x, fun, jac):
    result = OptimizeResult(
        x=x,
        fun=fun,
        jac=jac,
        nit=1,
        nfev=3,
        njev=3,
        success=True,
        status=0,
        message="CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH",
    )

    normalized = _opt_ref._normalize_scipy_result(result, x_dtype=np.float32)

    assert normalized.success is False
    assert normalized.status == 6
    assert "non-finite objective, iterate, or gradient" in normalized.message.lower()


def test_normalize_scipy_result_preserves_finite_success():
    result = OptimizeResult(
        x=np.asarray([0.5, 1.0], dtype=np.float64),
        fun=3.4,
        jac=np.asarray([0.1, 0.0], dtype=np.float64),
        nit=1,
        nfev=3,
        njev=3,
        success=True,
        status=0,
        message="CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH",
    )

    normalized = _opt_ref._normalize_scipy_result(result, x_dtype=np.float32)

    assert normalized.success is True
    assert normalized.status == 0
    assert normalized.message == "CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH"


def test_scipy_progress_callback_uses_latest_exact_duplicate_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluation_count = 0
    progress_events: list[tuple[int, float, float]] = []

    def value_and_grad(x: ArrayLike) -> tuple[np.float64, _HostVector]:
        nonlocal evaluation_count
        evaluation_count += 1
        x_host = np.asarray(x, dtype=np.float64)
        return np.float64(evaluation_count), np.full_like(x_host, evaluation_count)

    def fake_scipy_minimize(
        fun: _ScipyObjective,
        x0: _HostVector,
        *,
        jac: bool,
        method: str,
        options: dict[str, float | int],
        callback: _ScipyCallback | None,
    ) -> OptimizeResult:
        del method, options
        assert jac is True
        assert callback is not None
        fun(x0)
        accepted = x0 + 0.5
        fun(accepted)
        accepted_fun, accepted_gradient = fun(accepted)
        fun(x0 + 1.0)
        callback(accepted)
        return OptimizeResult(
            x=accepted,
            fun=accepted_fun,
            jac=accepted_gradient,
            nit=1,
            nfev=4,
            njev=4,
            success=True,
            status=0,
            message="test convergence",
        )

    monkeypatch.setattr(_opt_ref, "scipy_minimize", fake_scipy_minimize)

    def progress_callback(iteration: int, fun: float, grad_inf: float) -> None:
        progress_events.append((iteration, fun, grad_inf))

    result = _opt_ref.target_scipy_minimize_value_and_grad(
        value_and_grad,
        np.asarray([2.0, -1.0], dtype=np.float64),
        method="lbfgs",
        tol=1.0e-10,
        maxiter=1,
        options={"progress_callback": progress_callback},
    )

    assert result.nfev == evaluation_count == 4
    assert progress_events == [(1, 3.0, 3.0)]


def test_scipy_progress_callback_uses_exact_earlier_evaluation_without_reevaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    objective_points: list[np.ndarray] = []
    progress_events: list[tuple[int, float, float]] = []

    def value_and_grad(x: ArrayLike) -> tuple[np.float64, _HostVector]:
        x_host = np.asarray(x, dtype=np.float64)
        objective_points.append(x_host.copy())
        return np.float64(np.dot(x_host, x_host)), 2.0 * x_host

    def progress_callback(iteration: int, fun: float, grad_inf: float) -> None:
        progress_events.append((iteration, fun, grad_inf))

    def fake_scipy_minimize(
        fun: _ScipyObjective,
        x0: _HostVector,
        *,
        jac: bool,
        method: str,
        options: dict[str, float | int],
        callback: _ScipyCallback | None,
    ) -> OptimizeResult:
        del method, options
        assert jac is True
        assert callback is not None
        fun(x0)
        fun(x0 + np.asarray([0.25, -0.25], dtype=np.float64))
        first_accepted = x0 + np.asarray([0.5, -0.5], dtype=np.float64)
        fun(first_accepted)
        fun(x0 + np.asarray([0.75, -0.75], dtype=np.float64))
        callback(first_accepted)
        second_accepted = x0 + np.asarray([1.0, -1.0], dtype=np.float64)
        second_fun, second_grad = fun(second_accepted)
        callback(second_accepted)
        return OptimizeResult(
            x=second_accepted,
            fun=second_fun,
            jac=second_grad,
            nit=2,
            nfev=5,
            njev=5,
            success=True,
            status=0,
            message="test convergence",
        )

    monkeypatch.setattr(_opt_ref, "scipy_minimize", fake_scipy_minimize)
    result = _opt_ref.target_scipy_minimize_value_and_grad(
        value_and_grad,
        np.asarray([2.0, -1.0], dtype=np.float64),
        method="lbfgs",
        tol=1.0e-10,
        maxiter=2,
        options={"progress_callback": progress_callback},
    )

    assert result.nfev == len(objective_points) == 5
    assert progress_events == [
        (1, 8.5, 5.0),
        (2, 13.0, 6.0),
    ]


def test_scipy_state_and_progress_callbacks_each_run_once_per_accepted_iterate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    callback_points: list[np.ndarray] = []
    progress_events: list[tuple[int, float, float]] = []
    callback_order: list[tuple[str, int]] = []

    def value_and_grad(x: ArrayLike) -> tuple[np.float64, _HostVector]:
        x_host = np.asarray(x, dtype=np.float64)
        return np.float64(np.dot(x_host, x_host)), 2.0 * x_host

    def state_callback(x: ArrayLike) -> None:
        callback_points.append(np.asarray(x, dtype=np.float64))
        callback_order.append(("state", len(callback_points)))

    def progress_callback(iteration: int, fun: float, grad_inf: float) -> None:
        progress_events.append((iteration, fun, grad_inf))
        callback_order.append(("progress", iteration))

    def fake_scipy_minimize(
        fun: _ScipyObjective,
        x0: _HostVector,
        *,
        jac: bool,
        method: str,
        options: dict[str, float | int],
        callback: _ScipyCallback | None,
    ) -> OptimizeResult:
        del method, options
        assert jac is True
        assert callback is not None
        fun(x0)
        first_accepted = np.asarray([1.5, -0.5], dtype=np.float64)
        fun(first_accepted)
        callback(first_accepted)
        second_accepted = np.asarray([0.5, -0.25], dtype=np.float64)
        final_fun, final_grad = fun(second_accepted)
        callback(second_accepted)
        return OptimizeResult(
            x=second_accepted,
            fun=final_fun,
            jac=final_grad,
            nit=2,
            nfev=3,
            njev=3,
            success=True,
            status=0,
            message="test convergence",
        )

    monkeypatch.setattr(_opt_ref, "scipy_minimize", fake_scipy_minimize)

    _opt_ref.target_scipy_minimize_value_and_grad(
        value_and_grad,
        np.asarray([2.0, -1.0], dtype=np.float64),
        method="lbfgs",
        tol=1.0e-10,
        maxiter=2,
        options={
            "callback": state_callback,
            "progress_callback": progress_callback,
        },
    )

    np.testing.assert_array_equal(
        callback_points,
        np.asarray([[1.5, -0.5], [0.5, -0.25]], dtype=np.float64),
    )
    assert progress_events == [
        (1, 2.5, 3.0),
        (2, 0.3125, 1.0),
    ]
    assert callback_order == [
        ("state", 1),
        ("progress", 1),
        ("state", 2),
        ("progress", 2),
    ]


def test_scipy_state_callback_uses_recorded_evaluation_without_progress_observer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    callback_points: list[np.ndarray] = []

    def value_and_grad(x: ArrayLike) -> tuple[np.float64, _HostVector]:
        x_host = np.asarray(x, dtype=np.float64)
        return np.float64(np.dot(x_host, x_host)), 2.0 * x_host

    def fake_scipy_minimize(
        fun: _ScipyObjective,
        x0: _HostVector,
        *,
        jac: bool,
        method: str,
        options: dict[str, float | int],
        callback: _ScipyCallback | None,
    ) -> OptimizeResult:
        del method, options
        assert jac is True
        assert callback is not None
        fun(x0)
        accepted = x0 + 0.5
        accepted_fun, accepted_gradient = fun(accepted)
        callback(accepted)
        return OptimizeResult(
            x=accepted,
            fun=accepted_fun,
            jac=accepted_gradient,
            nit=1,
            nfev=2,
            njev=2,
            success=True,
            status=0,
            message="test convergence",
        )

    monkeypatch.setattr(_opt_ref, "scipy_minimize", fake_scipy_minimize)

    def state_callback(x: ArrayLike) -> None:
        callback_points.append(np.asarray(x, dtype=np.float64).copy())

    _opt_ref.target_scipy_minimize_value_and_grad(
        value_and_grad,
        np.asarray([2.0, -1.0], dtype=np.float64),
        method="lbfgs",
        tol=1.0e-10,
        maxiter=1,
        options={"callback": state_callback},
    )

    np.testing.assert_array_equal(
        callback_points,
        np.asarray([[2.5, -0.5]], dtype=np.float64),
    )


def test_scipy_progress_callback_rejects_metrics_from_a_different_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def value_and_grad(x: ArrayLike) -> tuple[np.float64, _HostVector]:
        x_host = np.asarray(x, dtype=np.float64)
        return np.float64(np.dot(x_host, x_host)), 2.0 * x_host

    def fake_scipy_minimize(
        fun: _ScipyObjective,
        x0: _HostVector,
        *,
        jac: bool,
        method: str,
        options: dict[str, float | int],
        callback: _ScipyCallback | None,
    ) -> OptimizeResult:
        del method, options
        assert jac is True
        assert callback is not None
        fun(x0)
        callback(x0 + 1.0)
        raise AssertionError("accepted-iterate mismatch must fail before return")

    monkeypatch.setattr(_opt_ref, "scipy_minimize", fake_scipy_minimize)

    def ignore_progress(iteration: int, fun: float, grad_inf: float) -> None:
        del iteration, fun, grad_inf

    with pytest.raises(RuntimeError, match="no exact objective evaluation"):
        _opt_ref.target_scipy_minimize_value_and_grad(
            value_and_grad,
            np.asarray([2.0, -1.0], dtype=np.float64),
            method="lbfgs",
            tol=1.0e-10,
            maxiter=2,
            options={"progress_callback": ignore_progress},
        )


def test_scipy_objective_trace_has_global_evaluation_ordinals_and_legacy_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def value_and_grad(x: ArrayLike) -> tuple[np.float64, _HostVector]:
        x_host = np.asarray(x, dtype=np.float64)
        return np.float64(np.dot(x_host, x_host)), 2.0 * x_host

    def fake_scipy_minimize(
        fun: _ScipyObjective,
        x0: _HostVector,
        *,
        jac: bool,
        method: str,
        options: dict[str, float | int],
        callback: _ScipyCallback | None,
    ) -> OptimizeResult:
        del method, options
        assert jac is True
        assert callback is None
        fun(x0)
        final_x = x0 + 0.5
        final_fun, final_grad = fun(final_x)
        return OptimizeResult(
            x=final_x,
            fun=final_fun,
            jac=final_grad,
            nit=1,
            nfev=2,
            njev=2,
            success=True,
            status=0,
            message="test convergence",
        )

    monkeypatch.setattr(_opt_ref, "scipy_minimize", fake_scipy_minimize)
    result = _opt_ref.target_scipy_minimize_value_and_grad(
        value_and_grad,
        np.asarray([2.0, -1.0], dtype=np.float64),
        method="lbfgs",
        tol=1.0e-10,
        maxiter=1,
        options={"record_scipy_callback_trace": True},
    )

    assert result.scipy_objective_evaluation_trace is result.scipy_callback_trace
    assert [
        event["objective_evaluation_index"]
        for event in result.scipy_objective_evaluation_trace
    ] == [1, 2]
    assert all(
        "line_search_evaluation" not in event
        for event in result.scipy_objective_evaluation_trace
    )


@pytest.mark.parametrize("method", ["bfgs", "lbfgs"])
def test_scipy_progress_observer_preserves_real_optimizer_trajectory(
    method: str,
) -> None:
    def rosenbrock_value_and_grad(
        x: ArrayLike,
    ) -> tuple[np.float64, _HostVector]:
        x_host = np.asarray(x, dtype=np.float64)
        x0, x1 = x_host
        residual = x1 - x0 * x0
        value = 100.0 * residual * residual + (1.0 - x0) ** 2
        gradient = np.asarray(
            [
                -400.0 * x0 * residual - 2.0 * (1.0 - x0),
                200.0 * residual,
            ],
            dtype=np.float64,
        )
        return np.float64(value), gradient

    x0 = np.asarray([-1.2, 1.0], dtype=np.float64)
    without_progress = _opt_ref.target_scipy_minimize_value_and_grad(
        rosenbrock_value_and_grad,
        x0,
        method=method,
        tol=1.0e-12,
        maxiter=5,
        options={"ftol": 0.0, "maxls": 20},
    )
    progress_events: list[tuple[int, float, float]] = []

    def progress_callback(iteration: int, fun: float, grad_inf: float) -> None:
        progress_events.append((iteration, fun, grad_inf))

    with_progress = _opt_ref.target_scipy_minimize_value_and_grad(
        rosenbrock_value_and_grad,
        x0,
        method=method,
        tol=1.0e-12,
        maxiter=5,
        options={
            "ftol": 0.0,
            "maxls": 20,
            "progress_callback": progress_callback,
        },
    )

    assert with_progress.nit == without_progress.nit == len(progress_events)
    assert with_progress.nfev == without_progress.nfev
    assert with_progress.njev == without_progress.njev
    assert float(with_progress.fun) == float(without_progress.fun)
    np.testing.assert_array_equal(with_progress.x, without_progress.x)
    np.testing.assert_array_equal(with_progress.jac, without_progress.jac)
