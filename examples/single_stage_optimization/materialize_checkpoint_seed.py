from __future__ import annotations

import argparse
import json
from pathlib import Path

from import_provenance import configure_local_simsopt_imports

configure_local_simsopt_imports(__file__)

from banana_opt.checkpoint_seed_sidecar import (  # noqa: E402
    materialize_checkpoint_seed,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Promote a single-stage checkpoint into a standalone seed "
            "directory (biot_savart_opt.json + contract-only results.json) "
            "that the seed loaders accept. Prefers the checkpoint's emitted "
            "checkpoint_seed_sidecar.json (sha-verified); for historical "
            "checkpoints derives the sidecar from the parent run's "
            "results.json plus realized currents read from the checkpoint "
            "graph. Never writes into the source run tree."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        type=Path,
        help="Checkpoint directory containing biot_savart.json (checkpoint_iterNNNN).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help=(
            "Seed directory to create (refuses to overwrite an existing "
            "biot_savart_opt.json / results.json there)."
        ),
    )
    parser.add_argument(
        "--parent-results",
        type=Path,
        default=None,
        help=(
            "Parent run results.json for historical checkpoints without an "
            "emitted sidecar (default: <checkpoint-dir>/../results.json)."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_results_path = materialize_checkpoint_seed(
        args.checkpoint_dir,
        args.output_dir,
        parent_results_path=args.parent_results,
    )
    sidecar = json.loads(output_results_path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "results_path": str(output_results_path),
                "biot_savart_path": str(
                    output_results_path.with_name("biot_savart_opt.json")
                ),
                "STAGE2_BS_SHA256": sidecar.get("STAGE2_BS_SHA256"),
                "CHECKPOINT_ITERATION": sidecar.get("CHECKPOINT_ITERATION"),
                "CHECKPOINT_PARENT_OUT_DIR": sidecar.get("CHECKPOINT_PARENT_OUT_DIR"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
