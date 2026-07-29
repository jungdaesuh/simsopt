"""SSOT contract for the stochastic Stage-II mirror and parity builder."""

from __future__ import annotations

import ast
import hashlib
from dataclasses import asdict
from pathlib import Path

from examples.jax.parity.cases.native_stage_two_optimization_stochastic import (
    SURFACE_INPUT,
    _scale_configuration,
)
from simsopt_jax.examples import stochastic_stage_two_configuration


def test_parity_builder_uses_the_production_stochastic_configuration() -> None:
    for scale in ("bounded", "native_default"):
        expected = asdict(stochastic_stage_two_configuration(scale))
        expected["surface_input_sha256"] = hashlib.sha256(
            SURFACE_INPUT.read_bytes()
        ).hexdigest()

        assert _scale_configuration(scale) == expected


def test_stochastic_configuration_preserves_native_and_bounded_scales() -> None:
    bounded = stochastic_stage_two_configuration("bounded")
    native = stochastic_stage_two_configuration("native_default")

    assert (bounded.surface_nphi, bounded.surface_ntheta) == (4, 4)
    assert (native.surface_nphi, native.surface_ntheta) == (64, 16)
    assert (bounded.training_sample_count, bounded.out_of_sample_count) == (2, 4)
    assert (native.training_sample_count, native.out_of_sample_count) == (16, 256)
    assert (bounded.training_seed, bounded.out_of_sample_seed) == (0, 1)
    assert (bounded.max_steps, native.max_steps) == (20, 400)


def test_stochastic_example_defaults_are_owned_by_the_shared_configuration() -> None:
    module = ast.parse(
        (
            Path(__file__).resolve().parents[2]
            / "examples/jax/2_Intermediate/stage_two_optimization_stochastic.py"
        ).read_text(encoding="utf-8")
    )
    run_example_call = next(
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_example"
    )
    keywords = {
        keyword.arg: keyword.value
        for keyword in run_example_call.keywords
        if keyword.arg is not None
    }

    def configuration_scale(node: ast.expr) -> str | None:
        if not (
            isinstance(node, ast.Attribute)
            and node.attr == "max_steps"
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "stochastic_stage_two_configuration"
            and len(node.value.args) == 1
            and isinstance(node.value.args[0], ast.Constant)
            and isinstance(node.value.args[0].value, str)
        ):
            return None
        return node.value.args[0].value

    assert configuration_scale(keywords["bounded_steps"]) == "bounded"
    assert configuration_scale(keywords["native_default_steps"]) == "native_default"
