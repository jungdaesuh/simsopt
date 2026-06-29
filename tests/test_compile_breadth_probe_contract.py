from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path

import pytest

from benchmarks import compile_breadth_probe as probe


def _probe_args(output_json: Path) -> Namespace:
    return Namespace(
        mpol_list=[6, 8, 10],
        warm_start_run_dir=None,
        stage2_bs_path="stage2.json",
        plasma_surf_filename="wout.nc",
        equilibria_dir="equilibria",
        vol_target=0.04,
        iota_target=0.285,
        num_tf_coils=20,
        output_json=str(output_json),
    )


def test_compile_breadth_checkpoint_write_preserves_previous_json_on_temp_write_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_json = tmp_path / "compile_breadth.json"
    output_json.write_text('{"status": "previous"}\n', encoding="utf-8")
    original_write_text = Path.write_text

    def interrupted_temp_write(
        path: Path,
        data: str,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> int:
        if path == output_json:
            raise AssertionError("checkpoint writer must not overwrite target directly")
        if path.name.startswith(f".{output_json.name}.") and path.name.endswith(".tmp"):
            raise OSError("interrupted temp write")
        return original_write_text(
            path,
            data,
            encoding=encoding,
            errors=errors,
            newline=newline,
        )

    monkeypatch.setattr(Path, "write_text", interrupted_temp_write)

    with pytest.raises(OSError, match="interrupted temp write"):
        probe._write_json(output_json, {"status": "running"})

    assert json.loads(output_json.read_text(encoding="utf-8")) == {"status": "previous"}


def test_compile_breadth_checkpoint_preserves_partial_sweep_artifact(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("JAX_LOG_COMPILES", "1")
    monkeypatch.setenv("XLA_FLAGS", "--xla_dump_to=/tmp/cbp_hlo")
    output_json = tmp_path / "compile_breadth.json"
    args = _probe_args(output_json)
    resolutions = [
        probe._Resolution(mpol=6, ntor=6, nphi=79, ntheta=24),
        probe._Resolution(mpol=8, ntor=8, nphi=103, ntheta=28),
    ]
    completed_record = {
        "mpol": 6,
        "ntor": 6,
        "nphi": 79,
        "ntheta": 24,
        "seed_state": "fresh-solve",
        "kernels": {},
    }

    def fake_probe_resolution(
        parsed_args: Namespace,
        resolution: probe._Resolution,
        seed_resolved: object,
    ) -> dict[str, object]:
        assert parsed_args is args
        assert seed_resolved is None
        if resolution.mpol == 6:
            return completed_record
        raise RuntimeError("mpol8 compile failed")

    monkeypatch.setattr(probe, "_parse_args", lambda: args)
    monkeypatch.setattr(probe, "_resolutions", lambda parsed_args: resolutions)
    monkeypatch.setattr(probe, "_seed_resolved_surface", lambda parsed_args: None)
    monkeypatch.setattr(probe, "_probe_resolution", fake_probe_resolution)

    with pytest.raises(RuntimeError, match="mpol8 compile failed"):
        probe.main()

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["active_resolution"] == {
        "mpol": 8,
        "ntor": 8,
        "nphi": 103,
        "ntheta": 28,
    }
    assert payload["completed_mpol"] == [6]
    assert payload["results_by_mpol"]["6"] == completed_record
    assert payload["compile_environment"] == {
        "JAX_LOG_COMPILES": "1",
        "XLA_FLAGS": "--xla_dump_to=/tmp/cbp_hlo",
    }
    assert payload["case"]["mpol_list"] == [6, 8, 10]
    assert payload["failure"] == {
        "exception_type": "RuntimeError",
        "message": "mpol8 compile failed",
    }
