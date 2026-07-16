"""Static subprocess probe for full-loop interpreter preservation."""

from __future__ import annotations

from collections.abc import Sequence
import json
import sys

from import_smoke_cases import prefer_local_simsopt_source_tree


prefer_local_simsopt_source_tree()

from benchmarks.single_stage_full_loop_compare import (  # noqa: E402
    _normalize_args,
    _parser,
    build_lane_command,
)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    if arguments == ("--print-prefix",):
        print(sys.prefix)
        return 0

    args = _parser().parse_args(arguments)
    _normalize_args(args)
    commands = {
        lane: build_lane_command(
            args,
            lane=lane,
            run_dir=args.output_root / lane,
            run_config_sha256="d" * 64,
        )[0]
        for lane in ("cpu", "jax")
    }
    print(
        json.dumps(
            {
                "commands": commands,
                "normalized_python": str(args.python),
                "sys_prefix": sys.prefix,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
