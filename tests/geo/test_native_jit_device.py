"""Native ``simsopt.geo.jit`` and the JAX devices native geometry runs on.

``JitOwnershipTests`` needs only JAX, as native simsopt does: the wrapper's
input ownership, JAX transformation compatibility and device routing.
``TestNativeJaxDevice`` needs the optional JAX runtime: backend validation,
the x64 and platform settings importing ``simsopt.geo`` makes, and native
geometry next to ``set_backend``, mostly in fresh processes
(``fresh_process/native_import.py`` and ``fresh_process/set_backend_startup.py``)
because JAX picks its platforms once, when it starts.
"""

import os
import subprocess
import sys
import unittest
from collections import OrderedDict, UserDict, defaultdict, namedtuple
from contextlib import nullcontext
from pathlib import Path
from unittest import mock

from unittest_jax_support import (
    JaxTestCase,
    aligned_host_copy,
    optional_jax_imports,
    require_cuda,
)

with optional_jax_imports():
    from simsopt_jax.backend import set_backend

import jax
import jax.numpy as jnp
import numpy as np
import simsopt.geo.jit as jit_module
from simsopt.geo.config import parameters
from simsopt.geo.jit import jit


Arrays = namedtuple("Arrays", ["first", "rest"])


class HashableArray(np.ndarray):
    __eq__ = object.__eq__

    # A deliberately hashable ndarray fixture exercises JAX static arguments.
    def __hash__(self) -> int:
        return object.__hash__(self)


def _static_array():
    constant = np.array([3.0]).view(HashableArray)
    constant.flags.writeable = False
    return constant


def _positional_adder(traced):
    def add_static(value, constant):
        traced.append(constant)
        return value + constant[0]

    return add_static


def _variadic_adder(traced):
    def add_last(value, *constants):
        traced.append(constants[-1])
        return value + constants[0] + constants[-1][0]

    return add_last


def _keyword_only_adder(traced):
    def add_static(value, *, options):
        traced.append(options)
        return value + options[0][0]

    return add_static


def positional(wrapped, static):
    return wrapped(2.0, static)


def keyword(wrapped, static):
    return wrapped(2.0, constant=static)


class IndexSelector:
    def __index__(self):
        return 1


class FrozenDictionary(dict):
    setter_calls = 0

    def __setitem__(self, key, value):
        self.setter_calls += 1
        raise TypeError("FrozenDictionary does not allow item assignment")


class TransformDictionary(dict):
    setter_calls = 0

    def __setitem__(self, key, value):
        self.setter_calls += 1
        super().__setitem__(key, 2 * value)


@jax.tree_util.register_pytree_node_class
class RegisteredArrays:
    def __init__(self, values):
        self.values = values

    def tree_flatten(self):
        return (self.values,), None

    @classmethod
    def tree_unflatten(cls, metadata, children):
        return cls(children[0])


class JitOwnershipTests(unittest.TestCase):
    def test_no_jit_snapshots_nested_inputs_and_retains_plain_execution(self):
        """With jit off, bare and nested arrays are copied, the call runs on the native device, and the name is kept."""

        def record(x, *, nested):
            return x, nested

        with mock.patch.dict(parameters, {"jit": False}):
            wrapped = jit_module.jit(record)
        x, y = np.ones(3), np.ones(4)
        device = object()
        with mock.patch.object(jit_module, "native_jax_device", return_value=device), \
                mock.patch.object(jit_module.jax, "default_device") as context:
            held, nested = wrapped(x, nested={"arrays": [y]})
        x[:] = 4
        y[:] = 7
        np.testing.assert_array_equal(held, np.ones(3))
        np.testing.assert_array_equal(nested["arrays"][0], np.ones(4))
        context.assert_called_once_with(device)
        self.assertEqual(wrapped.__name__, record.__name__)

    def test_compiled_dispatch_receives_owned_inputs_under_native_device_context(self):
        """Both resolved selectors reach jax.jit, so no JAX version infers either;
        a negative index names its parameter and counts from the signature."""
        cases = (
            ((1,), lambda wrapped, x, static: wrapped(x, static)),
            (-1, lambda wrapped, x, static: wrapped(x, marker=static)),
        )
        for argnums, call in cases:
            with self.subTest(static_argnums=argnums):
                device_sentinel = object()
                static_sentinel = object()
                x = np.ones(4)
                fake_compile = mock.Mock(side_effect=lambda fun, **kwargs: fun)
                context = mock.Mock(side_effect=lambda device: nullcontext())
                with mock.patch.dict(parameters, {"jit": True}), \
                        mock.patch.object(jit_module, "jaxjit", fake_compile), \
                        mock.patch.object(jit_module, "native_jax_device", return_value=device_sentinel), \
                        mock.patch.object(jit_module.jax, "default_device", context):
                    wrapped = jit_module.jit(lambda a, marker: (a, marker), static_argnums=argnums)
                    held, marker = call(wrapped, x, static_sentinel)
                x[:] = 5
                np.testing.assert_array_equal(held, np.ones(4))
                self.assertIs(marker, static_sentinel)
                context.assert_called_once_with(device_sentinel)
                self.assertEqual(
                    fake_compile.call_args.kwargs,
                    {"static_argnums": (1,), "static_argnames": ("marker",)},
                )

    def test_real_jax_outputs_keep_call_time_values_after_input_mutation(self):
        """Even ready identity outputs of bare, nested and keyword arrays own their values."""

        def identities(values, bundle, *, other):
            return jnp.asarray(values), jnp.asarray(bundle[0]["arrays"][0]), jnp.asarray(other["value"])

        for use_jit in (True, False):
            with self.subTest(jit=use_jit), mock.patch.dict(parameters, {"jit": use_jit}):
                wrapped = jit_module.jit(identities)
                inputs = [aligned_host_copy(np.arange(1.0, n + 1.0)) for n in (16, 8, 16)]
                expected = [values.copy() for values in inputs]
                held = jax.block_until_ready(wrapped(inputs[0], [{"arrays": (inputs[1],)}], other={"value": inputs[2]}))
                for values in inputs:
                    values[:] = -7.0
                for actual, values in zip(held, expected):
                    np.testing.assert_array_equal(actual, values)
                self.assertEqual(held[0].devices(), {jit_module.native_jax_device()})

    def test_real_jax_static_arguments_and_wraps_metadata_are_preserved(self):
        """Positional and keyword statics compile on the native device, and the wrapper keeps the function metadata."""

        def powers(values, order, *, scale):
            """Scaled powers with positional and keyword static parameters."""
            return scale * jnp.sum(values**order)

        with mock.patch.dict(parameters, {"jit": True}):
            wrapped = jit_module.jit(powers, static_argnums=(1,), static_argnames=("scale",))
        values = np.array([1.0, 2.0, 3.0])
        result = wrapped(values, 3, scale=0.5)
        self.assertEqual(float(result), 18.0)
        self.assertEqual(result.devices(), {jit_module.native_jax_device()})
        self.assertIs(wrapped.__wrapped__, powers)
        self.assertEqual(wrapped.__name__, powers.__name__)
        self.assertEqual(wrapped.__doc__, powers.__doc__)

    def test_tracers_grad_hessian_nested_jit_and_vmap_preserve_derivatives(self):
        """grad, Hessian, raw jax.jit and vmap through the wrapper match analytic derivatives, with jit on and off."""
        values = np.linspace(0.25, 1.0, 4)
        for use_jit in (True, False):
            with self.subTest(jit=use_jit), mock.patch.dict(parameters, {"jit": use_jit}):
                wrapped = jit_module.jit(lambda x: jnp.sum(x**3))
                np.testing.assert_allclose(jax.grad(wrapped)(values), 3 * values**2, rtol=1e-14, atol=0)
                np.testing.assert_allclose(jax.jacfwd(jax.grad(wrapped))(values), np.diag(6 * values), rtol=1e-14, atol=0)
                nested = jax.jit(jax.grad(wrapped))  # noqa: TID251 checks composition with raw jax.jit
                np.testing.assert_allclose(nested(values), 3 * values**2, rtol=1e-14, atol=0)
                batch = np.stack([values, 2 * values])
                np.testing.assert_allclose(jax.vmap(wrapped)(batch), np.sum(batch**3, axis=1), rtol=1e-14, atol=0)

    def test_plain_builtin_mappings_preserve_key_order_and_factory(self):
        """With jit off, copied dict, OrderedDict and defaultdict inputs keep their type, key order and factory."""
        mappings: list[dict[object, np.ndarray | list[float]]] = [
            {1: np.array([2.0]), "a": np.array([3.0])},
            {"z": np.array([2.0]), "a": np.array([3.0])},
            OrderedDict([("z", np.array([2.0])), ("a", np.array([3.0]))]),
            defaultdict(list, {1: np.array([2.0]), "a": np.array([3.0])}),
            defaultdict(list, {"z": np.array([2.0]), "a": np.array([3.0])}),
        ]
        with mock.patch.dict(parameters, {"jit": False}):
            identity = jit(lambda values: values)
        for mapping in mappings:
            with self.subTest(kind=type(mapping), keys=tuple(mapping)):
                nested = Arrays([mapping], None)
                copied = identity(nested).first[0]
                self.assertIs(type(copied), type(mapping))
                self.assertEqual(tuple(copied), tuple(mapping))
                for key in mapping:
                    self.assertFalse(np.shares_memory(copied[key], mapping[key]))
                    np.testing.assert_array_equal(copied[key], mapping[key])
        self.assertIs(identity(mappings[-1]).default_factory, list)

    def test_explicit_statics_without_signature(self):
        """Explicit static selectors compile builtins such as getattr and min, which have no signature."""
        with mock.patch.dict(parameters, {"jit": True}):
            attribute = jit(getattr, static_argnums=1)
            minimum = jit(min, static_argnums=(0, 1))
        self.assertEqual(tuple(int(item) for item in attribute(jnp.array([3.0, 4.0]), "shape")), (2,))
        self.assertEqual(int(minimum(3, 2)), 2)

    def test_unregistered_dictionaries_are_opaque_leaves_without_hooks(self):
        """With jit off, dictionaries JAX does not register are returned as is and their item setters never run."""
        with mock.patch.dict(parameters, {"jit": False}):
            wrapped = jit(lambda values: values)
        for kind in (FrozenDictionary, TransformDictionary, UserDict):
            with self.subTest(kind=kind.__name__):
                incoming = kind({"entry": np.array([3.0])})
                self.assertIs(wrapped(incoming), incoming)
                self.assertIs(wrapped(incoming)["entry"], incoming["entry"])
                np.testing.assert_array_equal(wrapped(incoming)["entry"], [3.0])
                self.assertEqual(getattr(incoming, "setter_calls", 0), 0)

    def test_registered_custom_pytree_snapshots_dynamic_numpy_leaves(self):
        """NumPy leaves of a registered custom pytree are copied at call time, with jit on and off."""
        for use_jit in (False, True):
            with self.subTest(jit=use_jit), mock.patch.dict(parameters, {"jit": use_jit}):
                wrapped = jit(lambda container: jnp.asarray(container.values))
                incoming = RegisteredArrays(aligned_host_copy(np.arange(1.0, 17.0)))
                expected = incoming.values.copy()
                held = wrapped(incoming).block_until_ready()
                incoming.values[:] = -7
                np.testing.assert_array_equal(held, expected)

    def test_matrix_semantics_and_storage_with_jit_off(self):
        """With jit off, an np.matrix input is copied and keeps its type and matrix product."""
        with mock.patch.dict(parameters, {"jit": False}):
            square = jit(lambda value: (value, value * value))
        matrix = np.matrix([[1.0, 2.0], [3.0, 4.0]])
        copied, result = square(matrix)
        self.assertIsInstance(copied, np.matrix)
        self.assertIsInstance(result, np.matrix)
        self.assertFalse(np.shares_memory(copied, matrix))
        matrix[:] = -7
        np.testing.assert_array_equal(copied, [[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_array_equal(result, [[7.0, 10.0], [15.0, 22.0]])

    def test_mask_semantics_and_storage_with_jit_off(self):
        """With jit off, a masked-array input is copied with its mask, and the result still ignores masked entries."""
        with mock.patch.dict(parameters, {"jit": False}):
            square = jit(lambda value: (value, value * value))
        masked = np.ma.array([2.0, 9.0], mask=[False, True])
        copied, result = square(masked)
        self.assertIsInstance(copied, np.ma.MaskedArray)
        self.assertIsInstance(result, np.ma.MaskedArray)
        self.assertFalse(np.shares_memory(copied.data, masked.data))
        self.assertFalse(np.shares_memory(copied.mask, masked.mask))
        masked.data[:] = -7
        np.ma.getmaskarray(masked)[:] = False
        np.testing.assert_array_equal(copied.data, [2.0, 9.0])
        np.testing.assert_array_equal(copied.mask, [False, True])
        np.testing.assert_array_equal(result.mask, [False, True])
        self.assertEqual(result.sum(), 4.0)

    def assert_traced_once(self, make, options, call, static, expected=5.0, is_static=True):
        """Three calls trace once; ``is_static`` says whether tracing saw ``static`` itself."""
        traced = []
        with mock.patch.dict(parameters, {"jit": True}):
            wrapped = jit(make(traced), **options)
        for _ in range(3):
            self.assertEqual(float(call(wrapped, static)), expected)
        self.assertEqual(len(traced), 1)
        self.assertEqual(traced[0] is static, is_static)

    def test_static_selectors_keep_identity_cache_and_actual_call_positioning(self):
        """Static NumPy objects keep identity and cache hits; explicit selectors
        do not infer the other selector or bind keyword calls."""
        inferred = ({"static_argnums": 1}, {"static_argnums": (1,)}, {"static_argnums": np.int64(1)},
                    {"static_argnums": np.array(1, dtype=np.int64)}, {"static_argnums": IndexSelector()},
                    {"static_argnums": -1}, {"static_argnames": "constant"})
        cases = [
            *((options, call, True) for options in inferred for call in (positional, keyword)),
            ({"static_argnums": 1, "static_argnames": ()}, positional, True),
            ({"static_argnums": 1, "static_argnames": ()}, keyword, False),
            ({"static_argnums": (), "static_argnames": "constant"}, positional, False),
            ({"static_argnums": (), "static_argnames": "constant"}, keyword, True),
            ({"static_argnums": iter((1,))}, positional, True),
            ({"static_argnames": iter(("constant",))}, positional, True),
        ]
        for options, call, is_static in cases:
            with self.subTest(options=options, call=call.__name__):
                self.assert_traced_once(_positional_adder, options, call, _static_array(), is_static=is_static)
        with self.subTest("negative index in a variadic call"):
            self.assert_traced_once(_variadic_adder, {"static_argnums": -1},
                                    lambda wrapped, static: wrapped(2.0, 4.0, static), _static_array(), expected=9.0)
        with self.subTest("keyword-only static subtree is not copied"):
            self.assert_traced_once(_keyword_only_adder, {"static_argnames": ("options",)},
                                    lambda wrapped, static: wrapped(2.0, options=static), (_static_array(),))

    def test_jax_input_donation_is_preserved(self):
        """donate_argnums still donates a JAX array input, which is deleted after the call."""
        with mock.patch.dict(parameters, {"jit": True}):
            wrapped = jit(lambda value: value + 1, donate_argnums=0)
        value = jax.device_put(np.ones(3), jax.devices("cpu")[0])  # noqa: TID251 test places on a chosen device
        np.testing.assert_array_equal(wrapped(value), np.full(3, 2.0))
        self.assertTrue(value.is_deleted())


def _fresh_environment():
    return {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("JAX_")
    }


_FRESH_PROCESS = Path(__file__).resolve().with_name("fresh_process")


def _run_fresh_process(script, *arguments, environment):
    """Run ``fresh_process/<script>`` in a new interpreter; it fails on its asserts."""
    subprocess.run(
        [sys.executable, str(_FRESH_PROCESS / script), *arguments],
        env=environment,
        check=True,
    )


class TestNativeJaxDevice(JaxTestCase):
    def test_unknown_backend_is_rejected(self):
        """An unknown backend is rejected, naming the accepted calls."""
        with self.assertRaisesRegex(
            ValueError,
            r"set_backend\('jax_cpu_unknown', device=None\) is not valid; use set_backend\('native'\)",
        ):
            # The name is deliberately outside the accepted backend values.
            set_backend("jax_cpu_unknown")  # pyright: ignore[reportArgumentType]

    def test_native_import_preserves_platform_selection_in_fresh_process(self):
        """Importing native geometry respects explicit CPU platform selection and enables
        double precision."""
        for platform_variable in [None, "JAX_PLATFORMS", "JAX_PLATFORM_NAME"]:
            with self.subTest(platform_variable=platform_variable), self.case():
                self._case_native_import_preserves_platform_selection_in_fresh_process(
                    platform_variable
                )

    def _case_native_import_preserves_platform_selection_in_fresh_process(
        self, platform_variable
    ):
        """Run one row with case-local objects released before runtime cleanup."""
        environment = _fresh_environment()
        if platform_variable is not None:
            environment[platform_variable] = "cpu"
        _run_fresh_process("native_import.py", "platform_selection", environment=environment)

    def test_native_import_enables_x64_despite_explicit_disable_in_fresh_process(self):
        """Importing native geometry restores double precision even when the environment
        explicitly disables it."""
        for x64_setting in ["false", "0"]:
            with self.subTest(x64_setting=x64_setting), self.case():
                self._case_native_import_enables_x64_despite_explicit_disable_in_fresh_process(
                    x64_setting
                )

    def _case_native_import_enables_x64_despite_explicit_disable_in_fresh_process(
        self, x64_setting
    ):
        """Run one row with case-local objects released before runtime cleanup."""
        environment = _fresh_environment()
        environment["JAX_PLATFORMS"] = "cpu"
        environment["JAX_ENABLE_X64"] = x64_setting
        _run_fresh_process("native_import.py", "x64", environment=environment)

    def test_native_jax_curve_constructed_before_set_backend_in_fresh_process(self):
        """A helical curve constructed before backend setup still yields finite float64
        length and field values."""
        _run_fresh_process(
            "set_backend_startup.py",
            "curve_before_set_backend",
            environment=_fresh_environment(),
        )

    def test_set_backend_default_platform_after_native_import_in_fresh_process(self):
        """``set_backend`` makes its platform the default after ``simsopt.geo`` set ``cpu``.

        Without JAX platform variables, ``simsopt.geo`` sets the legacy
        ``jax_platform_name="cpu"``, which JAX consults before ``jax_platforms``.
        Every row checks that ``set_backend`` clears it and that its platform is
        then the default backend. The CUDA rows (skipped without a CUDA device)
        are the failures the fix removed: ``Unknown backend cpu`` or, with
        ``cuda,cpu`` exported after the import, CPU while CUDA was requested.
        """
        for device, platforms_after_import in [
            ("cpu", None),
            ("gpu", None),
            ("gpu", "cuda,cpu"),
        ]:
            with self.subTest(
                device=device, platforms_after_import=platforms_after_import
            ), self.case():
                self._case_set_backend_default_platform_after_native_import_in_fresh_process(
                    device, platforms_after_import
                )

    def _case_set_backend_default_platform_after_native_import_in_fresh_process(
        self, device, platforms_after_import
    ):
        """Run one row with case-local objects released before runtime cleanup."""
        if device == "gpu":
            require_cuda("CUDA JAX device required")
        _run_fresh_process(
            "set_backend_startup.py",
            "after_native_import",
            device,
            platforms_after_import or "",
            environment=_fresh_environment(),
        )

    def test_native_stage_two_with_explicit_platforms_in_fresh_process(self):
        """Native stage II after ``set_backend``; ``gpu-no-platform-env`` exports no
        JAX platform variable, so ``simsopt.geo`` sets ``cpu`` before ``set_backend``."""
        for case_id_0, (device, jax_platforms) in zip(
            ["cpu", "gpu-cuda-cpu-platforms", "gpu-no-platform-env"],
            [("cpu", "cpu"), ("gpu", "cuda,cpu"), ("gpu", None)],
            strict=True,
        ):
            with self.subTest(
                case_id_0=case_id_0, device=device, jax_platforms=jax_platforms
            ), self.case():
                self._case_native_stage_two_with_explicit_platforms_in_fresh_process(
                    device, jax_platforms
                )

    def _case_native_stage_two_with_explicit_platforms_in_fresh_process(
        self, device, jax_platforms
    ):
        """Run one row with case-local objects released before runtime cleanup."""
        if device == "gpu":
            require_cuda("CUDA JAX device required")
        environment = _fresh_environment()
        if jax_platforms is not None:
            environment["JAX_PLATFORMS"] = jax_platforms
        _run_fresh_process(
            "set_backend_startup.py",
            "stage_two",
            device,
            jax_platforms or "",
            environment=environment,
        )

    def _run_platform_startup(self, scenario, jax_platforms, *, needs_cuda):
        """Run one ``set_backend_startup.py`` scenario in a fresh process."""
        if needs_cuda:
            require_cuda("CUDA JAX device required")
        environment = _fresh_environment()
        if jax_platforms is not None:
            environment["JAX_PLATFORMS"] = jax_platforms
        _run_fresh_process(
            "set_backend_startup.py", scenario, environment=environment
        )

    def test_gpu_selected_at_startup_also_starts_the_cpu_in_fresh_process(self):
        """Without ``JAX_PLATFORMS``, a GPU selection starts CUDA as default and the CPU for native kernels."""
        self._run_platform_startup("gpu_at_startup", None, needs_cuda=True)

    def test_selections_after_initialization_keep_the_platforms_in_fresh_process(self):
        """Under ``cuda,cpu``, CPU-then-GPU selections after initialization keep native kernels and platforms."""
        self._run_platform_startup(
            "cpu_then_gpu_after_initialization", "cuda,cpu", needs_cuda=True
        )

    def test_explicit_cuda_platforms_start_cuda_alone_in_fresh_process(self):
        """An explicit ``JAX_PLATFORMS=cuda`` starts only CUDA, and native kernels fall back to it."""
        self._run_platform_startup("explicit_cuda_only", "cuda", needs_cuda=True)

    def test_cpu_selected_before_initialization_keeps_cuda_off_in_fresh_process(self):
        """A CPU selection before initialization starts only the CPU, also under ``cuda,cpu``."""
        for jax_platforms in (None, "cuda,cpu"):
            with self.subTest(jax_platforms=jax_platforms), self.case():
                self._run_platform_startup(
                    "cpu_before_initialization", jax_platforms, needs_cuda=False
                )

    def test_native_jit_runs_on_cpu_with_cuda_default_in_fresh_process(self):
        """Under ``JAX_PLATFORMS=cuda,cpu`` native kernels return CPU arrays,
        also with jit disabled (``SIMSGEOJIT=0``)."""
        require_cuda("CUDA JAX device required")
        for use_jit in ("1", "0"):
            with self.subTest(SIMSGEOJIT=use_jit):
                environment = _fresh_environment()
                environment.update(JAX_PLATFORMS="cuda,cpu", SIMSGEOJIT=use_jit)
                _run_fresh_process(
                    "native_import.py", "jit_device_policy", environment=environment
                )


if __name__ == "__main__":
    unittest.main()
