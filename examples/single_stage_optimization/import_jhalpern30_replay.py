from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from banana_opt.jhalpern30_compat import import_jhalpern30_stage_bundle


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Import a historical jhalpern30 stageNN bsurf_opt.json/state.json "
            "bundle into a current-repo replay artifact directory."
        ),
    )
    parser.add_argument(
        "bundle_root",
        help="Directory containing stageNN/bsurf_opt.json and stageNN/state.json.",
    )
    parser.add_argument(
        "--stage-name",
        default=os.environ.get("JHALPERN30_STAGE_NAME", "stage00"),
        help="Historical stage directory to import, default stage00.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where biot_savart_opt.json and results.json are written.",
    )
    parser.add_argument(
        "--plasma-surf-path",
        required=True,
        help=(
            "WOUT file used to stamp WOUT_CONVENTION/WOUT_OFF_SPEC on the "
            "current-repo seed artifact."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    imported_bs_path, imported_results_path = import_jhalpern30_stage_bundle(
        args.bundle_root,
        args.output_dir,
        plasma_surf_path=args.plasma_surf_path,
        stage_name=args.stage_name,
    )
    print(f"Imported BiotSavart: {imported_bs_path}")
    print(f"Imported results: {imported_results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
