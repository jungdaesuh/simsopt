from __future__ import annotations

import argparse
import json
from pathlib import Path

from import_provenance import configure_local_simsopt_imports

configure_local_simsopt_imports(__file__)

from banana_opt.finite_current_materializer import (  # noqa: E402
    FINITE_CURRENT_FIELD_SOURCE_MODES,
    MaterializedFiniteCurrentSeedRequest,
    materialize_finite_current_seed,
)
from banana_opt.finite_current_profiles import (  # noqa: E402
    format_proxy_current_sign_convention_help,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize proxy plasma-current and VF coils into a "
            "Stage-2-compatible BiotSavart seed.\n"
            "Retargeting current requires materializing a new seed;\n"
            "single-stage replay only loads persisted field sources."
        ),
        epilog=format_proxy_current_sign_convention_help(
            FINITE_CURRENT_FIELD_SOURCE_MODES,
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source-biot-savart",
        required=True,
        type=Path,
        help="Existing Stage-2 or replay-compatible BiotSavart seed.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="Directory where biot_savart_opt.json and results.json are written.",
    )
    parser.add_argument(
        "--finite-current-mode",
        required=True,
        choices=FINITE_CURRENT_FIELD_SOURCE_MODES,
        help="Finite-current proxy/VF field-source profile to materialize.",
    )
    parser.add_argument(
        "--proxy-current-A",
        required=True,
        type=float,
        help=(
            "Proxy plasma-current scalar in amperes. Sign semantics are selected "
            "by --finite-current-mode; see the mode table below."
        ),
    )
    parser.add_argument(
        "--stage2-results",
        type=Path,
        default=None,
        help="Optional source results.json path; defaults to source sibling.",
    )
    parser.add_argument(
        "--equilibrium-file",
        type=Path,
        default=None,
        help=(
            "VMEC/wout file for Wataru proxy placement. Defaults to "
            "PLASMA_SURF_PATH from source results.json."
        ),
    )
    parser.add_argument(
        "--vf-template-path",
        type=Path,
        default=None,
        help="Optional VF template override. Defaults to the selected profile template.",
    )
    parser.add_argument(
        "--vf-current-A",
        type=float,
        default=None,
        help=(
            "Optional VF current scalar. Defaults and sign semantics are selected "
            "by --finite-current-mode; see the mode table below."
        ),
    )
    parser.add_argument(
        "--toroidal-flux",
        type=float,
        default=None,
        help=(
            "Normalized toroidal flux for Wataru proxy placement. Defaults to "
            "TOROIDAL_FLUX from source results.json."
        ),
    )
    parser.add_argument(
        "--surface-scale-factor",
        type=float,
        default=1.0,
        help="Surface-family scale factor for Wataru proxy placement.",
    )
    parser.add_argument(
        "--nphi",
        type=int,
        default=127,
        help="Toroidal quadrature count passed to the Wataru proxy builder.",
    )
    parser.add_argument(
        "--ntheta",
        type=int,
        default=32,
        help="Poloidal quadrature count passed to the Wataru proxy builder.",
    )
    parser.add_argument(
        "--num-tf-coils",
        type=int,
        default=20,
        help="Expected TF-coil count used to partition legacy source artifacts.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output biot_savart_opt.json/results.json pair.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = materialize_finite_current_seed(
        MaterializedFiniteCurrentSeedRequest(
            source_biot_savart=args.source_biot_savart,
            output_root=args.output_root,
            finite_current_mode=args.finite_current_mode,
            proxy_current_A=args.proxy_current_A,
            stage2_results=args.stage2_results,
            equilibrium_file=args.equilibrium_file,
            vf_template_path=args.vf_template_path,
            vf_current_A=args.vf_current_A,
            toroidal_flux=args.toroidal_flux,
            surface_scale_factor=args.surface_scale_factor,
            nphi=args.nphi,
            ntheta=args.ntheta,
            num_tf_coils=args.num_tf_coils,
            overwrite=args.overwrite,
        )
    )
    print(json.dumps(result.to_json_payload(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
