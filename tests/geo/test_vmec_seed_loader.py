from pathlib import Path
import sys

import numpy as np
import pytest


EXAMPLES_ROOT = Path(__file__).resolve().parents[2] / "examples" / "single_stage_optimization"
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

from banana_opt import vmec_seed_loader  # noqa: E402
from banana_opt.vmec_seed_loader import (  # noqa: E402
    _find_paired_input,
    materialize_rerunnable_input,
)


def test_find_paired_input_uses_vmec_naming_convention(tmp_path: Path) -> None:
    wout_path = tmp_path / "wout_s01_seed_opt.nc"
    input_path = tmp_path / "input.s01_seed_opt"
    wout_path.touch()
    input_path.touch()

    assert _find_paired_input(wout_path) == input_path


def test_find_paired_input_returns_none_for_unpaired_wout(tmp_path: Path) -> None:
    wout_path = tmp_path / "wout_s01_seed_opt.nc"
    wout_path.touch()

    assert _find_paired_input(wout_path) is None


def test_materialize_rejects_wout_without_paired_input(tmp_path: Path) -> None:
    wout_path = tmp_path / "wout_s01_seed_opt.nc"
    wout_path.touch()

    with pytest.raises(ValueError, match="wout alone"):
        materialize_rerunnable_input(
            wout_path,
            out_path=tmp_path / "input.s04_child",
        )


def test_materialize_rejects_stage_a_free_boundary_request(tmp_path: Path) -> None:
    wout_path = tmp_path / "wout_s01_seed_opt.nc"
    wout_path.touch()

    with pytest.raises(ValueError, match="fixed-boundary"):
        materialize_rerunnable_input(
            wout_path,
            out_path=tmp_path / "input.s04_child",
            lfreeb=True,
        )


def test_materialize_accepts_vmec_none_mgrid_sentinel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wout_path = tmp_path / "wout_s01_seed_opt.nc"
    input_path = tmp_path / "input.s01_seed_opt"
    out_path = tmp_path / "input.s04_child"
    wout_path.touch()
    input_path.touch()

    class FakeIndata:
        lfreeb = False
        pres_scale = 1.0
        curtor = 1.0
        mgrid_file = b"NONE"
        ac = np.ones(2)
        am = np.ones(2)

    class FakeVmec:
        def __init__(self, source: str, verbose: bool = False) -> None:
            self.source = source
            self.verbose = verbose
            self.indata = FakeIndata()

        def write_input(self, destination: str) -> None:
            Path(destination).write_text("materialized\n", encoding="utf-8")

    monkeypatch.setattr(vmec_seed_loader, "Vmec", FakeVmec)

    assert materialize_rerunnable_input(wout_path, out_path=out_path) == out_path
    assert out_path.read_text(encoding="utf-8") == "materialized\n"


def test_materialize_rejects_real_mgrid_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wout_path = tmp_path / "wout_s01_seed_opt.nc"
    input_path = tmp_path / "input.s01_seed_opt"
    wout_path.touch()
    input_path.touch()

    class FakeIndata:
        lfreeb = False
        mgrid_file = b"mgrid.nc"

    class FakeVmec:
        def __init__(self, source: str, verbose: bool = False) -> None:
            self.source = source
            self.verbose = verbose
            self.indata = FakeIndata()

    monkeypatch.setattr(vmec_seed_loader, "Vmec", FakeVmec)

    with pytest.raises(ValueError, match="populated mgrid_file"):
        materialize_rerunnable_input(
            wout_path,
            out_path=tmp_path / "input.s04_child",
        )
