"""Checkpoint seed sidecars: make single-stage checkpoints loadable as seeds.

A single-stage checkpoint directory historically held only ``biot_savart.json``
plus surface artifacts, while every seed loader hard-requires a sibling
``results.json`` (existence + contract keys + ``STAGE2_BS_SHA256`` content
binding) — so the program's "the checkpoint is the answer" invariant could not
be acted on without ad-hoc, fabrication-prone sidecar copies. This module is
the single owner of the seed-contract key set: the solver writes a
CONTRACT-ONLY sidecar (``checkpoint_seed_sidecar.json`` — a filename no
run-artifact scanner globs, so ``rglob("results.json")`` consumers never ingest
checkpoints as run rows) at checkpoint-save time, and
``materialize_checkpoint_seed`` promotes a checkpoint into a standalone seed
directory (``biot_savart_opt.json`` + ``results.json``) that
``load_single_stage_resume_seed_results`` / ``load_stage2_artifact_results``
accept. Sidecars NEVER carry realized run metrics (``MAX_CURVATURE``,
``FINAL_*``, ``J`` …): a checkpoint's metrics were never measured, and a
sidecar that invents them is exactly the fabrication the program brief
forbids. Copying is whitelist-only, so metrics cannot leak by construction.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Mapping

from .artifact_contracts import (
    STAGE2_BS_SHA256_KEY,
    compute_stage2_bs_sha256,
)
from .json_compat import load_boozer_finite_i
from .stage2_single_stage_handoff import partition_loaded_stage2_coils

CHECKPOINT_SEED_SIDECAR_FILENAME = "checkpoint_seed_sidecar.json"
CHECKPOINT_SEED_SCHEMA_VERSION = 1
CHECKPOINT_SEED_ARTIFACT_KIND = "single_stage_checkpoint_seed"
CHECKPOINT_SEED_NOTE = (
    "Contract-only checkpoint seed sidecar: carries seed-loading contract and "
    "provenance fields only; realized run metrics are deliberately absent "
    "(they were never measured for this checkpoint state)."
)

# Keys whose presence in the sidecar must come from the checkpoint's own
# biot_savart.json (its dofs/currents), never from the parent run. The builder
# rejects them in the parent-copy set so a stale parent value can never
# impersonate a realized checkpoint value.
_REALIZED_ONLY_KEYS = frozenset({STAGE2_BS_SHA256_KEY})

# Seed-contract keys copied verbatim from the parent run's loaded seed
# metadata when present. Every entry exists because a seed-loading consumer
# reads it:
#   MAJOR_RADIUS / TOROIDAL_FLUX   validate_single_stage_resume_seed_contract
#   banana_surf_radius             resolve_single_stage_banana_surf_radius
#   order / mpol / ntor            main-flow order resolution; thin-runner
#                                  seed mpol/ntor preflight
#   TF_CURRENT_A                   resolve_single_stage_resume_tf_current_A
#   NUM_TF_COILS / COIL_GROUPS / NUM_BANANA_COILS / NUM_PROXY_COILS /
#   NUM_VF_COILS                   _resolve_stage2_coil_manifest partition
#   FINITE_CURRENT_MODE family     finite-current mode lock + Boozer current
#   FLIP_BANANA / IOTA_TARGET_SIGN resolve_stage2_seed_flip_banana
#   BANANA_INIT_CURRENT_A /
#   BANANA_CURRENT_MAX_A           legacy banana-current metadata upgrade
#   CURVATURE_THRESHOLD            curvature hardware contract preflight
#   PLASMA_SURF_PATH / WOUT_* /
#   OFFSPEC / STRICT               lane-honesty + wout-convention preflight
# The coil-graph structure (counts, manifest) cannot change during a run, so
# parent values remain exactly true for every checkpoint of that run.
_SEED_CONTRACT_COPY_KEYS = (
    "MAJOR_RADIUS",
    "TOROIDAL_FLUX",
    "banana_surf_radius",
    "order",
    "mpol",
    "ntor",
    "TF_CURRENT_A",
    "NUM_TF_COILS",
    "COIL_GROUPS",
    "NUM_BANANA_COILS",
    "NUM_PROXY_COILS",
    "NUM_VF_COILS",
    "FINITE_CURRENT_MODE",
    "FINITE_CURRENT_MODE_SOURCE",
    "BOOZER_CURRENT_CONVENTION",
    "PLASMA_CURRENT_A",
    "BOOZER_I",
    "PROXY_PLASMA_CURRENT_A",
    "VF_CURRENT_A",
    "VF_TEMPLATE_PATH",
    "FLIP_BANANA",
    "IOTA_TARGET_SIGN",
    "BANANA_INIT_CURRENT_A",
    "BANANA_CURRENT_MAX_A",
    "BANANA_CURRENT_MODE",
    "CURVATURE_THRESHOLD",
    "PLASMA_SURF_PATH",
    "WOUT_CONVENTION",
    "WOUT_OFF_SPEC",
    "WOUT_RBTOR_SIGN",
    "WOUT_PHI_EDGE_SIGN",
    "WOUT_EXPECTED_LANE",
    "WOUT_CONVENTION_SOURCE_PATH",
    "OFFSPEC_REPLAY_DEBUG_ONLY",
    "STRICT_VACUUM_CURRENT",
    "ACCEPT_OFFSPEC_R0_SEED",
    # Realized embedded CWS winding torus (2026-06-10): the surface the banana
    # angle-dofs actually live on. Copy-if-present so re-centered/remapped
    # lineages stay self-describing through materialization.
    "BANANA_WINDING_SURFACE_MAJOR_RADIUS_M",
    "COIL_WINDING_SURFACE_MAJOR_RADIUS_M",
    "BANANA_CWS_EMBEDDED_WINDING_MINOR_RADIUS_M",
)

# Keys the resume loaders hard-require; the builder fails loud when the merged
# sidecar misses one, naming the consumer that would have rejected the seed.
_REQUIRED_SEED_KEYS = (
    ("MAJOR_RADIUS", "validate_single_stage_resume_seed_contract"),
    ("TOROIDAL_FLUX", "validate_single_stage_resume_seed_contract"),
    ("banana_surf_radius", "resolve_single_stage_banana_surf_radius"),
)

_CHECKPOINT_DIR_ITERATION_RE = re.compile(r"checkpoint_iter(\d+)$")


def bind_results_to_bs(results: dict, bs_path: str | Path) -> dict:
    """Stamp ``STAGE2_BS_SHA256`` into ``results``, binding it by content to
    the Biot-Savart artifact at ``bs_path``. Returns ``results``."""
    results[STAGE2_BS_SHA256_KEY] = compute_stage2_bs_sha256(bs_path)
    return results


def build_checkpoint_seed_sidecar(
    parent_seed_results: Mapping[str, object],
    *,
    checkpoint_bs_path: str | Path,
    checkpoint_iteration: int | None,
    parent_out_dir: str | Path,
    runtime_fields: Mapping[str, object] | None = None,
) -> dict:
    """Build the contract-only sidecar for one checkpoint Biot-Savart file.

    ``parent_seed_results`` is the parent run's loaded seed metadata (the
    sidecar the run itself was seeded from, or the run's own results.json for
    historical materialization); ``runtime_fields`` overlays run-true values
    (geometry, currents, lane flags) and wins over copied parent values.
    """
    sidecar: dict = {}
    for key in _SEED_CONTRACT_COPY_KEYS:
        value = parent_seed_results.get(key)
        if value is not None:
            sidecar[key] = value
    if "IOTA_TARGET_SIGN" in sidecar and "FLIP_BANANA" not in sidecar:
        raise ValueError(
            "Parent seed metadata has IOTA_TARGET_SIGN without FLIP_BANANA; "
            "resolve_stage2_seed_flip_banana would reject the resulting seed."
        )
    if runtime_fields:
        forbidden = _REALIZED_ONLY_KEYS.intersection(runtime_fields)
        if forbidden:
            raise ValueError(
                f"runtime_fields may not set realized-only keys: {sorted(forbidden)}; "
                "the checksum binding is computed from the checkpoint artifact itself."
            )
        sidecar.update(
            {key: value for key, value in runtime_fields.items() if value is not None}
        )
    missing = [
        (key, consumer)
        for key, consumer in _REQUIRED_SEED_KEYS
        if sidecar.get(key) is None
    ]
    if missing:
        raise ValueError(
            "Checkpoint seed sidecar is missing required seed-contract keys: "
            + "; ".join(f"{key} (required by {consumer})" for key, consumer in missing)
        )
    sidecar["ARTIFACT_KIND"] = CHECKPOINT_SEED_ARTIFACT_KIND
    sidecar["CHECKPOINT_SEED_SCHEMA_VERSION"] = CHECKPOINT_SEED_SCHEMA_VERSION
    sidecar["CHECKPOINT_SEED_NOTE"] = CHECKPOINT_SEED_NOTE
    sidecar["CHECKPOINT_ITERATION"] = (
        None if checkpoint_iteration is None else int(checkpoint_iteration)
    )
    sidecar["CHECKPOINT_PARENT_OUT_DIR"] = str(parent_out_dir)
    return bind_results_to_bs(sidecar, checkpoint_bs_path)


def write_checkpoint_seed_sidecar(
    checkpoint_dir: str | Path,
    parent_seed_results: Mapping[str, object],
    *,
    checkpoint_iteration: int | None,
    parent_out_dir: str | Path,
    runtime_fields: Mapping[str, object] | None = None,
) -> Path:
    """Write ``checkpoint_seed_sidecar.json`` next to the checkpoint's
    ``biot_savart.json`` (which must already be saved). Returns the path."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_bs_path = checkpoint_dir / "biot_savart.json"
    if not checkpoint_bs_path.is_file():
        raise ValueError(
            f"Checkpoint Biot-Savart artifact missing: {checkpoint_bs_path}; "
            "write the sidecar after bs.save()."
        )
    sidecar = build_checkpoint_seed_sidecar(
        parent_seed_results,
        checkpoint_bs_path=checkpoint_bs_path,
        checkpoint_iteration=checkpoint_iteration,
        parent_out_dir=parent_out_dir,
        runtime_fields=runtime_fields,
    )
    sidecar_path = checkpoint_dir / CHECKPOINT_SEED_SIDECAR_FILENAME
    sidecar_path.write_text(
        json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return sidecar_path


def _checkpoint_iteration_from_dir(checkpoint_dir: Path) -> int | None:
    match = _CHECKPOINT_DIR_ITERATION_RE.search(checkpoint_dir.name)
    return int(match.group(1)) if match else None


def _realized_banana_current_fields(
    checkpoint_bs_path: Path,
    parent_results: Mapping[str, object],
) -> dict:
    """Read the realized banana currents out of the checkpoint's serialized
    Biot-Savart graph (the honest per-checkpoint record the seed sign-contract
    preflight reads). Runs under the solver environment, exactly like the
    finite-current materializer."""
    bs = load_boozer_finite_i(str(checkpoint_bs_path))
    requested_num_tf_coils = int(parent_results.get("NUM_TF_COILS") or 20)
    partitions = partition_loaded_stage2_coils(
        bs.coils,
        stage2_results=parent_results,
        requested_num_tf_coils=requested_num_tf_coils,
    )
    realized_currents = [
        float(coil.current.get_value()) for coil in partitions.banana_coils
    ]
    return {
        "BANANA_CURRENTS_A": realized_currents,
        "NUM_TF_COILS": partitions.num_tf_coils,
        "NUM_BANANA_COILS": partitions.num_banana_coils,
        "NUM_PROXY_COILS": partitions.num_proxy_coils,
        "NUM_VF_COILS": partitions.num_vf_coils,
    }


def materialize_checkpoint_seed(
    checkpoint_dir: str | Path,
    output_dir: str | Path,
    *,
    parent_results_path: str | Path | None = None,
) -> Path:
    """Promote a checkpoint into a standalone seed directory the loaders accept.

    Copies ``biot_savart.json`` to ``<output_dir>/biot_savart_opt.json`` and
    writes the contract-only sidecar as ``<output_dir>/results.json``. Prefers
    the checkpoint's own ``checkpoint_seed_sidecar.json`` (emitted at save
    time; sha-verified against the artifact before reuse); for historical
    checkpoints it derives the sidecar from the parent run's ``results.json``
    plus the realized currents read from the checkpoint graph. Refuses to
    overwrite an existing seed artifact. Returns the written results.json path.
    """
    checkpoint_dir = Path(checkpoint_dir)
    output_dir = Path(output_dir)
    checkpoint_bs_path = checkpoint_dir / "biot_savart.json"
    if not checkpoint_bs_path.is_file():
        raise ValueError(f"Checkpoint Biot-Savart artifact missing: {checkpoint_bs_path}")
    output_bs_path = output_dir / "biot_savart_opt.json"
    output_results_path = output_dir / "results.json"
    if output_bs_path.exists() or output_results_path.exists():
        raise ValueError(
            f"Refusing to overwrite existing seed artifact under {output_dir}."
        )

    emitted_sidecar_path = checkpoint_dir / CHECKPOINT_SEED_SIDECAR_FILENAME
    if emitted_sidecar_path.is_file():
        sidecar = json.loads(emitted_sidecar_path.read_text(encoding="utf-8"))
        recorded_digest = sidecar.get(STAGE2_BS_SHA256_KEY)
        actual_digest = compute_stage2_bs_sha256(checkpoint_bs_path)
        if recorded_digest != actual_digest:
            raise ValueError(
                f"Checkpoint sidecar checksum mismatch: {emitted_sidecar_path} "
                f"reports {STAGE2_BS_SHA256_KEY}={recorded_digest!r}, but "
                f"{checkpoint_bs_path} hashes to {actual_digest!r}."
            )
    else:
        if parent_results_path is None:
            candidate = checkpoint_dir.parent / "results.json"
            if not candidate.is_file():
                raise ValueError(
                    "Historical checkpoint has no emitted sidecar and no parent "
                    f"results.json at {candidate}; pass parent_results_path."
                )
            parent_results_path = candidate
        parent_results = json.loads(
            Path(parent_results_path).read_text(encoding="utf-8")
        )
        if not isinstance(parent_results, dict):
            raise ValueError(
                f"Parent results.json must contain a JSON object: {parent_results_path}"
            )
        sidecar = build_checkpoint_seed_sidecar(
            parent_results,
            checkpoint_bs_path=checkpoint_bs_path,
            checkpoint_iteration=_checkpoint_iteration_from_dir(checkpoint_dir),
            parent_out_dir=checkpoint_dir.parent,
            runtime_fields=_realized_banana_current_fields(
                checkpoint_bs_path, parent_results
            ),
        )
        sidecar["CHECKPOINT_SEED_MATERIALIZED_FROM_PARENT_RESULTS"] = str(
            parent_results_path
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(checkpoint_bs_path, output_bs_path)
    # The copy is byte-identical, so the sidecar's recorded digest stays valid
    # for the promoted artifact; re-verify to fail loud on copy corruption.
    if compute_stage2_bs_sha256(output_bs_path) != sidecar[STAGE2_BS_SHA256_KEY]:
        raise ValueError(f"Seed copy corrupted: {output_bs_path} hash mismatch.")
    sidecar["CHECKPOINT_SEED_SOURCE_BS_PATH"] = str(checkpoint_bs_path)
    output_results_path.write_text(
        json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output_results_path
