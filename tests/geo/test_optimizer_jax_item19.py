from pathlib import Path

import pytest

import simsopt.geo.optimizer_jax as _opt


def test_item19_optimizer_jax_product_code_has_no_dynamic_private_import():
    source = Path(_opt.__file__).read_text()

    assert "import importlib" not in source
    assert "importlib.import_module" not in source
    assert "__import__(" not in source


def test_item19_private_package_loader_uses_fixed_import(monkeypatch):
    monkeypatch.setattr(_opt, "_private_pkg", None)

    pkg = _opt._load_private_pkg()

    assert pkg.__name__ == "simsopt.geo.optimizer_jax_private"
    assert _opt._load_private_pkg() is pkg


def test_item19_target_outer_loop_contract_defaults_to_ondevice_lbfgs():
    contract = _opt.resolve_target_outer_loop_optimizer_contract(
        "jax",
        "ondevice",
        component_label="item19 target optimizer",
    )

    assert contract == _opt.TargetOptimizerContract(
        method="lbfgs-ondevice",
        use_least_squares_objective=False,
    )


def test_item19_reference_and_target_optimizer_lanes_stay_explicit():
    reference_contract = _opt.resolve_reference_outer_loop_optimizer_contract(
        "cpu",
        "scipy",
        component_label="item19 reference optimizer",
    )
    assert reference_contract == _opt.ReferenceOptimizerContract(method="lbfgs")

    with pytest.raises(ValueError, match="requires optimizer_backend='ondevice'"):
        _opt.resolve_target_outer_loop_optimizer_contract(
            "jax",
            "scipy",
            component_label="item19 target optimizer",
        )

    with pytest.raises(ValueError, match="SciPy/reference optimizer lane"):
        _opt.resolve_reference_outer_loop_optimizer_contract(
            "jax",
            "ondevice",
            component_label="item19 reference optimizer",
        )
