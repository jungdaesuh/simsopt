"""The compiled B-VJP and dB/dX kernels keep fast per-coil reduction layouts.

Two XLA layout hazards made these kernels several times slower (openxla/xla
issue 52437):

* A per-coil reduce (over quadrature nodes, or over the point chunk in the
  VJP) whose operand keeps a length-3 Cartesian axis minor reduces the other
  axis at stride 3. The B and A integrands are component-major so that no
  per-node array carries that axis.
* On GPU, a reduce fused into a fusion whose root is a concatenate computes
  each output element's sum in a serial loop. Contracting each component with
  the currents before stacking the three keeps the stack out of the reduction
  fusions.

These tests compile both kernels for a small case and inspect the optimized
HLO text for either form.
"""

from __future__ import annotations

from unittest_jax_support import JaxTestCase, optional_jax_imports, require_cuda

with optional_jax_imports():
    import simsopt_jax  # noqa: F401
    import jax  # noqa: F401
    import jax.numpy as jnp
    from simsopt_jax.core import biotsavart as biotsavart_core


import re
from dataclasses import dataclass

import numpy as np

# Five coils is the only length-5 axis: 24 nodes (padded to 32 by the pairwise
# sum), 12-node blocks (padded to 16), 7-point chunks with a 6-point tail of 20
# points, and 3 Cartesian components.
_COIL_COUNT = 5
_QUADRATURE_COUNT = 24
_POINT_COUNT = 20
_POINT_CHUNK_SIZE = 7
# (coil_chunk_size, quadrature_block_size, point_chunk_size): unblocked, and
# two quadrature blocks as in the fast-lane tuning (128 nodes in 64-node blocks).
_TUNINGS = ((0, 0, _POINT_CHUNK_SIZE), (0, 12, _POINT_CHUNK_SIZE))

_COMPUTATION = re.compile(r"^(?:ENTRY\s+)?%([\w.\-]+)\s.*\{\s*$")
_INSTRUCTION = re.compile(
    r"^\s*(?P<root>ROOT\s+)?%(?P<name>[\w.\-]+)\s*=\s*"
    r"(?P<shape>\([^=]*?\)|\S+)\s+(?P<opcode>[\w\-]+)\((?P<rest>.*)$"
)
_ARRAY_SHAPE = re.compile(r"^[a-z]\w*\[(?P<dims>[\d,]*)\](?:\{(?P<layout>[\d,]*))?")
_CALLED = re.compile(r"\bcalls=%([\w.\-]+)")
_REDUCED_DIMENSIONS = re.compile(r"dimensions=\{([\d,]*)\}")


@dataclass(frozen=True)
class _Instruction:
    name: str
    shape: str
    opcode: str
    operands: tuple[str, ...]
    attributes: str
    is_root: bool


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
                is_root=instruction.group("root") is not None,
            )
        )
    return computations


@dataclass(frozen=True)
class _Reduce:
    operand_shape: str
    dims: tuple[int, ...]
    minor: int
    reduced: frozenset[int]

    def keeps_minor_component_axis(self):
        return self.dims[self.minor] == 3 and self.minor not in self.reduced

    def __str__(self):
        return f"{self.operand_shape} dimensions={sorted(self.reduced)}"


def _array_dims_and_minor(shape):
    """Return the dimensions and the layout-minor axis of an array shape."""
    ((dims_text, layout),) = _ARRAY_SHAPE.findall(shape)
    dims = tuple(int(size) for size in dims_text.split(",") if size)
    minor = int(layout.split(",")[0]) if layout else len(dims) - 1
    return dims, minor


def _per_coil_reduces(computations, coil_count):
    """Return every per-coil reduce of the module.

    A per-coil reduce keeps the coil axis: it sums a per-coil-node array over
    quadrature nodes, or over the point chunk in the VJP. Reductions over the
    coil axis (the current contraction) are not per-coil.
    """
    shapes = {
        instruction.name: instruction.shape
        for instructions in computations.values()
        for instruction in instructions
    }
    reduces = []
    for instructions in computations.values():
        for instruction in instructions:
            if instruction.opcode != "reduce":
                continue
            operand_shape = shapes[instruction.operands[0]]
            dims, minor = _array_dims_and_minor(operand_shape)
            (axes,) = _REDUCED_DIMENSIONS.findall(instruction.attributes)
            reduced = frozenset(int(axis) for axis in axes.split(",") if axis)
            if any(
                size == coil_count and axis not in reduced
                for axis, size in enumerate(dims)
            ):
                reduces.append(_Reduce(operand_shape, dims, minor, reduced))
    return reduces


def _computation_has_reduce(computations, name, seen=frozenset()):
    """Whether computation ``name`` or a fusion it calls contains a reduce."""
    for instruction in computations[name]:
        if instruction.opcode == "reduce":
            return True
        for called in _CALLED.findall(instruction.attributes):
            if called not in seen and _computation_has_reduce(
                computations, called, seen | {name}
            ):
                return True
    return False


def _concatenate_fusions_with_reduces(computations):
    """Return the fused computations rooted at a concatenate that reduce."""
    fused = {
        called
        for instructions in computations.values()
        for instruction in instructions
        if instruction.opcode == "fusion"
        for called in _CALLED.findall(instruction.attributes)
    }
    return sorted(
        name
        for name in fused
        if any(
            instruction.is_root and instruction.opcode == "concatenate"
            for instruction in computations[name]
        )
        and _computation_has_reduce(computations, name)
    )


def _inputs():
    rng = np.random.default_rng(52437)
    gammas = rng.standard_normal((_COIL_COUNT, _QUADRATURE_COUNT, 3))
    gammadashs = rng.standard_normal((_COIL_COUNT, _QUADRATURE_COUNT, 3))
    currents = rng.uniform(0.5, 1.5, size=_COIL_COUNT)
    points = rng.standard_normal((_POINT_COUNT, 3)) + 5.0
    cotangent = rng.standard_normal((_POINT_COUNT, 3))
    return tuple(
        jnp.asarray(values)
        for values in (points, cotangent, gammas, gammadashs, currents)
    )


def _optimized_hlo(kernel_name, tuning):
    """Compile one kernel for the default device and return its optimized HLO."""
    points, cotangent, gammas, gammadashs, currents = _inputs()
    if kernel_name == "B_vjp":
        kernel = biotsavart_core._make_B_vjp_kernel(*tuning)
        arguments = (points, cotangent, gammas, gammadashs, currents)
    else:
        kernel = biotsavart_core._make_kernel(
            biotsavart_core._Integrand.B,
            biotsavart_core._DiffMode.JACOBIAN,
            *tuning,
        )
        arguments = (points, gammas, gammadashs, currents)
    return _parse_hlo(kernel.lower(*arguments).compile().as_text())


_KERNELS = ("B_vjp", "dB_by_dX")


class TestBiotSavartReduceLayout(JaxTestCase):
    def test_per_coil_reduces_keep_no_minor_component_axis(self):
        """No per-coil reduce of the compiled B-VJP or dB/dX kernel keeps a
        length-3 component axis minor."""
        for kernel_name in _KERNELS:
            for tuning in _TUNINGS:
                with self.subTest(kernel=kernel_name, tuning=tuning), self.case():
                    self._case_per_coil_reduces_keep_no_minor_component_axis(
                        kernel_name, tuning
                    )

    def _case_per_coil_reduces_keep_no_minor_component_axis(self, kernel_name, tuning):
        """Run one row with case-local objects released before runtime cleanup."""
        reduces = _per_coil_reduces(_optimized_hlo(kernel_name, tuning), _COIL_COUNT)
        # The guard is vacuous if the parser finds no per-coil reduce at all.
        self.assertTrue(reduces, "no per-coil reduce found in the optimized HLO")
        offending = [
            str(reduce) for reduce in reduces if reduce.keeps_minor_component_axis()
        ]
        self.assertEqual(offending, [], "per-coil reduces with a minor 3-axis")

    def test_gpu_reduces_are_not_fused_into_concatenates(self):
        """On GPU no reduce of the compiled B-VJP or dB/dX kernel sits in a
        fusion rooted at a concatenate."""
        require_cuda(
            "a reduce inside a concatenate-rooted fusion is a GPU emitter hazard "
            "(the A100 input_concatenate_fusion); it needs a CUDA device"
        )
        for kernel_name in _KERNELS:
            for tuning in _TUNINGS:
                with self.subTest(kernel=kernel_name, tuning=tuning), self.case():
                    self._case_gpu_reduces_are_not_fused_into_concatenates(
                        kernel_name, tuning
                    )

    def _case_gpu_reduces_are_not_fused_into_concatenates(self, kernel_name, tuning):
        """Run one row with case-local objects released before runtime cleanup."""
        computations = _optimized_hlo(kernel_name, tuning)
        self.assertTrue(
            _per_coil_reduces(computations, _COIL_COUNT),
            "no per-coil reduce found in the optimized HLO",
        )
        self.assertEqual(_concatenate_fusions_with_reduces(computations), [])
