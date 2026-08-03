"""Run-id regression tests for direct driver inputs.

The parent artifact ids identify the upstream state, but each driver also
consumes local configuration values while constructing its objective. Every
such value must remain in that stage's registry whitelist.
"""

from copy import deepcopy
from pathlib import Path
import sys

import pytest
import yaml


HBT_DIR = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "single_stage_optimization"
    / "HBT_BANANA"
)
sys.path.insert(0, str(HBT_DIR / "utils"))

from run_registry import build_input_blob, compute_run_id  # noqa: E402


CONFIG = yaml.safe_load((HBT_DIR / "config.yaml").read_text())
COMMIT = "review-registry-test"


def _set_path(config, path, value):
    node = config
    segments = path.split(".")
    for segment in segments[:-1]:
        node = node[segment]
    node[segments[-1]] = value


def _run_id(stage, config):
    extra = {
        "stage1_id": "s01_parent",
        "stage2_id": "s02_parent",
    }
    blob = build_input_blob(stage, config, extra=extra)
    return compute_run_id(stage, blob, COMMIT)[0]


@pytest.mark.parametrize(
    ("stage", "path", "replacement"),
    [
        ("stage2", "banana_coils.current_cap_stage2", True),
        ("stage2", "tf_coils.current", -81_000.0),
        ("stage2", "tf_coils.num", 21),
        ("stage2", "stage2_optimizer.tol", 2.0e-15),
        ("singlestage", "device.nfp", 3),
        ("singlestage", "device.stellsym", False),
        ("singlestage", "tf_coils.current", -81_000.0),
        ("singlestage", "tf_coils.num", 21),
        ("singlestage", "banana_coils.curv_p", 3),
        ("singlestage", "singlestage_optimizer.tol", 2.0e-15),
    ],
)
def test_direct_driver_input_changes_run_id(stage, path, replacement):
    baseline = _run_id(stage, CONFIG)
    changed = deepcopy(CONFIG)
    _set_path(changed, path, replacement)

    assert _run_id(stage, changed) != baseline, (
        f"editing {path} must invalidate the {stage} run id"
    )


@pytest.mark.parametrize(
    ("driver", "stage"),
    [
        ("02_stage2_driver.py", "stage2"),
        ("03_singlestage_driver.py", "singlestage"),
    ],
)
def test_driver_installs_atexit_handler_before_run_directory_setup(driver, stage):
    source = (HBT_DIR / driver).read_text()

    assert source.index(f"registry.register_{stage}") < source.index(
        f'install_atexit_handler(registry, "{stage}", RUN_ID)'
    ) < source.index(f'RUN_DIR = run_dir("{stage}",')
