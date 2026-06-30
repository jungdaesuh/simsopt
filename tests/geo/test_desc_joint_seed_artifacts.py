import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = REPO_ROOT / "examples" / "single_stage_optimization"
EXAMPLES_ROOT_STR = str(EXAMPLES_ROOT)
if EXAMPLES_ROOT_STR not in sys.path:
    sys.path.insert(0, EXAMPLES_ROOT_STR)

from banana_opt.desc_joint_seed_manifest import (  # noqa: E402
    DESC_JOINT_SEED_MANIFEST_SCHEMA_VERSION,
    load_desc_joint_seed_manifest,
)

M36_ROOT = Path(
    "/Users/suhjungdae/code/columbia/simsopt-pr-jax-port-clean/"
    ".m18-adjoint-artifacts/homotopy_m36"
)
RUNS_ROOT = Path(
    "/Users/suhjungdae/code/columbia/autoresearch/campaigns/"
    "balance_pareto_singlestage_2026-06-17/runs"
)
SLID_ROOT = RUNS_ROOT / "slid_clean_R0p9095_2026-06-22"
GALLERY_ROOT = RUNS_ROOT / "poincare_default_gallery_2026-06-25"
IOTA_R0935_ROOT = RUNS_ROOT / "iota011_a150_R0935_2026-06-25"
IOTA_R0976_ROOT = RUNS_ROOT / "iota011_a150_R0976_2026-06-25"
IOTA_INWARD_ROOT = RUNS_ROOT / "iota011_inward_probe_2026-06-24"
WINDING_ROOT = RUNS_ROOT / "winding_candidates_2026-06-25"


def _candidate(
    label: str,
    group: str,
    surface: Path,
    field: Path,
    surface_kind: str,
    *,
    state: Path | None = None,
    poincare_metrics: Path | None = None,
    poincare_png: Path | None = None,
) -> dict[str, str]:
    payload = {
        "label": label,
        "group": group,
        "surface": str(surface),
        "field": str(field),
        "surface_kind": surface_kind,
    }
    if state is not None:
        payload["state"] = str(state)
    if poincare_metrics is not None:
        payload["poincare_metrics"] = str(poincare_metrics)
    if poincare_png is not None:
        payload["poincare_png"] = str(poincare_png)
    return payload


def _gallery_metrics(label: str) -> Path:
    return GALLERY_ROOT / f"PoincareDefault_{label}.json"


def _gallery_png(label: str) -> Path:
    return GALLERY_ROOT / f"PoincareDefault_{label}.png"


def _provided_seed_candidates() -> list[dict[str, str]]:
    return [
        _candidate(
            "m36_baseline",
            "m36",
            M36_ROOT / "baseline_m36_converged.json",
            SLID_ROOT / "slid_cws_field.json",
            "bare_surface",
            state=M36_ROOT / "baseline_m36_converged_state.json",
            poincare_metrics=_gallery_metrics("m36_baseline"),
            poincare_png=_gallery_png("m36_baseline"),
        ),
        _candidate(
            "m36_chomp",
            "m36",
            M36_ROOT / "chomp_m36_converged.json",
            SLID_ROOT / "slid_cws_field_chomp.json",
            "bare_surface",
            state=M36_ROOT / "chomp_m36_converged_state.json",
            poincare_metrics=_gallery_metrics("m36_chomp"),
            poincare_png=_gallery_png("m36_chomp"),
        ),
        _candidate(
            "m36_clean2mm",
            "m36",
            M36_ROOT / "clean2mm_m36_converged.json",
            SLID_ROOT / "slid_cws_field_clean2mm.json",
            "bare_surface",
            state=M36_ROOT / "clean2mm_m36_converged_state.json",
            poincare_metrics=_gallery_metrics("m36_clean2mm"),
            poincare_png=_gallery_png("m36_clean2mm"),
        ),
        _candidate(
            "m36_clean2p5mm",
            "m36",
            M36_ROOT / "clean2p5mm_m36_converged.json",
            SLID_ROOT / "slid_cws_field_clean2p5mm.json",
            "bare_surface",
            state=M36_ROOT / "clean2p5mm_m36_converged_state.json",
            poincare_metrics=_gallery_metrics("m36_clean2p5mm"),
            poincare_png=_gallery_png("m36_clean2p5mm"),
        ),
        _candidate(
            "slidclean_opt",
            "slid_clean",
            SLID_ROOT / "surf_opt_boozer_surface.json",
            SLID_ROOT / "biot_savart_opt.json",
            "boozer_surface",
            poincare_metrics=_gallery_metrics("slidclean_opt"),
            poincare_png=_gallery_png("slidclean_opt"),
        ),
        _candidate(
            "slidclean_chomp",
            "slid_clean",
            SLID_ROOT / "surf_chomp_boozer_surface.json",
            SLID_ROOT / "slid_cws_field_chomp.json",
            "boozer_surface",
            state=SLID_ROOT / "surf_chomp_boozer_state.json",
            poincare_metrics=_gallery_metrics("slidclean_chomp"),
            poincare_png=_gallery_png("slidclean_chomp"),
        ),
        _candidate(
            "slidclean_baseline",
            "slid_clean",
            SLID_ROOT / "surf_clean_baseline_boozer_surface.json",
            SLID_ROOT / "slid_cws_field.json",
            "boozer_surface",
            poincare_metrics=_gallery_metrics("slidclean_baseline"),
            poincare_png=_gallery_png("slidclean_baseline"),
        ),
        _candidate(
            "slidclean_clean2mm",
            "slid_clean",
            SLID_ROOT / "surf_clean2mm_boozer_surface.json",
            SLID_ROOT / "slid_cws_field_clean2mm.json",
            "boozer_surface",
            poincare_metrics=_gallery_metrics("slidclean_clean2mm"),
            poincare_png=_gallery_png("slidclean_clean2mm"),
        ),
        _candidate(
            "slidclean_clean2p5mm",
            "slid_clean",
            SLID_ROOT / "surf_clean2p5mm_boozer_surface.json",
            SLID_ROOT / "slid_cws_field_clean2p5mm.json",
            "boozer_surface",
            poincare_metrics=_gallery_metrics("slidclean_clean2p5mm"),
            poincare_png=_gallery_png("slidclean_clean2p5mm"),
        ),
        _candidate(
            "iota011_R0935",
            "iota",
            IOTA_R0935_ROOT / "surf_iota011_boozer_surface.json",
            IOTA_R0935_ROOT / "biot_savart_opt.json",
            "boozer_surface",
            state=IOTA_R0935_ROOT / "surf_iota011_boozer_state.json",
            poincare_metrics=IOTA_R0935_ROOT / "PoincareMetrics_opt_default.json",
            poincare_png=IOTA_R0935_ROOT / "PoincarePlot_opt_default.png",
        ),
        _candidate(
            "iota011_R0976",
            "iota",
            IOTA_R0976_ROOT / "surf_iota011_boozer_surface.json",
            IOTA_R0976_ROOT / "biot_savart_opt.json",
            "boozer_surface",
            state=IOTA_R0976_ROOT / "surf_iota011_boozer_state.json",
            poincare_metrics=IOTA_R0976_ROOT / "PoincareMetrics_opt_default.json",
            poincare_png=IOTA_R0976_ROOT / "PoincarePlot_opt_default.png",
        ),
        _candidate(
            "iota011_inward_probe",
            "iota",
            IOTA_INWARD_ROOT / "surf_iota011_boozer_surface.json",
            IOTA_INWARD_ROOT / "biot_savart_opt.json",
            "boozer_surface",
            state=IOTA_INWARD_ROOT / "surf_iota011_boozer_state.json",
            poincare_metrics=IOTA_INWARD_ROOT / "PoincareMetrics_opt_default.json",
            poincare_png=IOTA_INWARD_ROOT / "PoincarePlot_opt_default.png",
        ),
        _candidate(
            "winding_band0015_R0p92_a0p15",
            "winding",
            WINDING_ROOT
            / "band_cell0015_R0p92_a0p15_nearmiss"
            / "mpol=10-ntor=10-7c91b7d9"
            / "surf_opt_boozer_surface.json",
            WINDING_ROOT
            / "band_cell0015_R0p92_a0p15_nearmiss"
            / "mpol=10-ntor=10-7c91b7d9"
            / "biot_savart_opt.json",
            "boozer_surface",
            state=WINDING_ROOT
            / "band_cell0015_R0p92_a0p15_nearmiss"
            / "mpol=10-ntor=10-7c91b7d9"
            / "surf_opt_boozer_state.json",
            poincare_metrics=WINDING_ROOT
            / "band_cell0015_R0p92_a0p15_nearmiss"
            / "mpol=10-ntor=10-7c91b7d9"
            / "poincare_default_diagnostics.json",
            poincare_png=WINDING_ROOT
            / "band_cell0015_R0p92_a0p15_nearmiss"
            / "mpol=10-ntor=10-7c91b7d9"
            / "poincare_default_plot.png",
        ),
        _candidate(
            "winding_ext0009_R0p94_a0p15",
            "winding",
            WINDING_ROOT
            / "ext_cell0009_R0p94_a0p15_FULLPASS"
            / "mpol=10-ntor=10-4acbfd1a"
            / "surf_opt_boozer_surface.json",
            WINDING_ROOT
            / "ext_cell0009_R0p94_a0p15_FULLPASS"
            / "mpol=10-ntor=10-4acbfd1a"
            / "biot_savart_opt.json",
            "boozer_surface",
            state=WINDING_ROOT
            / "ext_cell0009_R0p94_a0p15_FULLPASS"
            / "mpol=10-ntor=10-4acbfd1a"
            / "surf_opt_boozer_state.json",
            poincare_metrics=WINDING_ROOT
            / "ext_cell0009_R0p94_a0p15_FULLPASS"
            / "mpol=10-ntor=10-4acbfd1a"
            / "poincare_default_diagnostics.json",
            poincare_png=WINDING_ROOT
            / "ext_cell0009_R0p94_a0p15_FULLPASS"
            / "mpol=10-ntor=10-4acbfd1a"
            / "poincare_default_plot.png",
        ),
        _candidate(
            "winding_ext0014_R0p95_a0p142",
            "winding",
            WINDING_ROOT
            / "ext_cell0014_R0p95_a0p142_FULLPASS"
            / "mpol=10-ntor=10-d75d90af"
            / "surf_opt_boozer_surface.json",
            WINDING_ROOT
            / "ext_cell0014_R0p95_a0p142_FULLPASS"
            / "mpol=10-ntor=10-d75d90af"
            / "biot_savart_opt.json",
            "boozer_surface",
            state=WINDING_ROOT
            / "ext_cell0014_R0p95_a0p142_FULLPASS"
            / "mpol=10-ntor=10-d75d90af"
            / "surf_opt_boozer_state.json",
            poincare_metrics=WINDING_ROOT
            / "ext_cell0014_R0p95_a0p142_FULLPASS"
            / "mpol=10-ntor=10-d75d90af"
            / "poincare_default_diagnostics.json",
            poincare_png=WINDING_ROOT
            / "ext_cell0014_R0p95_a0p142_FULLPASS"
            / "mpol=10-ntor=10-d75d90af"
            / "poincare_default_plot.png",
        ),
    ]


@pytest.mark.skipif(
    not M36_ROOT.exists() or not RUNS_ROOT.exists(),
    reason="provided local DESC-joint seed artifacts are not present on this machine",
)
def test_provided_poincare_seed_artifacts_load_as_desc_joint_manifest(tmp_path):
    manifest_path = tmp_path / "provided_seed_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": DESC_JOINT_SEED_MANIFEST_SCHEMA_VERSION,
                "candidates": _provided_seed_candidates(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    manifest = load_desc_joint_seed_manifest(manifest_path)

    assert len(manifest.candidates) == 15
    source_checksum_maps = [
        candidate.source_checksums()
        for candidate in manifest.candidates
    ]
    assert all(
        source_checksums["surface"] and source_checksums["field"]
        for source_checksums in source_checksum_maps
    )
    assert {
        candidate.label
        for candidate in manifest.candidates
        if candidate.surface_kind == "bare_surface"
    } == {
        "m36_baseline",
        "m36_chomp",
        "m36_clean2mm",
        "m36_clean2p5mm",
    }
    assert {
        candidate.label
        for candidate in manifest.candidates
        if candidate.surface_kind == "boozer_surface"
    } >= {
        "slidclean_chomp",
        "iota011_R0935",
        "winding_ext0009_R0p94_a0p15",
    }
