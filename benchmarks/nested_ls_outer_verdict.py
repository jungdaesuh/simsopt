"""Render the eight-term outer track's verdict section from its receipts.

Phase 5 item 3 of ``docs/nested_ls_upgrade_implementation_plan.md``. The
verdict is the last thing written and the easiest to get wrong, because by
the time it is written the numbers are hours old and the temptation is to
recall them rather than re-read them. This tool cannot recall anything: every
figure it prints is read out of a receipt on disk, and a missing or unreadable
receipt is a refusal, not a gap in the output.

It renders. It does not judge. Whether a measured speedup is worth claiming,
and under what scope, is a human decision recorded in the track doc; what this
guarantees is that the numbers the decision is made from are the numbers the
run actually produced.

    python benchmarks/nested_ls_outer_verdict.py \\
        --sweep docs/receipts/evidence/nested_ls_outer_native_omp_sweep_*.json \\
        --b3 docs/receipts/evidence/nested_ls_outer_b3_*.json \\
        --b37 docs/receipts/evidence/nested_ls_outer_b37_*.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Final

REPO = Path(__file__).resolve().parents[1]


class VerdictInputError(RuntimeError):
    """A receipt the verdict needs is absent, unreadable, or not a receipt."""


#: The key that identifies each receipt kind. A sweep artifact and a claim
#: receipt have different shapes, and an identity check that accepted either
#: shape for either role would let a sweep be passed as ``--b3`` -- which
#: reads fine and produces a verdict about nothing.
_RECEIPT_MARKER: Final[dict[str, str]] = {
    "sweep": "best_omp_num_threads",
    "B3": "claim_boundary",
    "B37": "claim_boundary",
}


def load_receipt(path: Path, *, label: str) -> dict[str, Any]:
    """Read one receipt, refusing anything that is not one OF ITS KIND.

    Fails closed on every failure mode separately, because "the verdict is
    missing a number" and "the verdict quoted a number from the wrong file"
    look identical downstream and only the first is survivable.

    The kind check is per-label rather than generic. A first draft accepted
    any receipt carrying ``claim_boundary`` or ``probe``, which rejected the
    real sweep artifact (it has neither) and would have accepted a sweep
    handed in as ``--b3``. Dry-running against the shipped receipts is what
    surfaced it.
    """

    if not path.is_file():
        raise VerdictInputError(f"{label} receipt does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise VerdictInputError(f"{label} receipt is unreadable: {path}: {error}")
    if not isinstance(payload, dict):
        raise VerdictInputError(f"{label} receipt is not a JSON object: {path}")
    marker = _RECEIPT_MARKER[label]
    if marker not in payload:
        raise VerdictInputError(
            f"{label} receipt has no {marker!r}; it is not a {label} receipt: {path}"
        )
    return payload


def require(payload: dict[str, Any], *keys: str, label: str) -> Any:
    """Read a nested key, naming the receipt and the path when it is absent."""

    node: Any = payload
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            raise VerdictInputError(f"{label} receipt has no {'.'.join(keys)}")
        node = node[key]
    return node


def _same_head(*payloads: tuple[str, dict[str, Any]]) -> str:
    """The single git_head every receipt must share, or a refusal.

    The claim driver already binds these pairwise; re-checking here is not
    redundant belt-and-braces, it is the one place a HUMAN reads all three at
    once. A verdict assembled from receipts taken at different commits would
    describe a program that never existed.
    """

    heads = {label: str(p.get("git_head", "")) for label, p in payloads}
    distinct = set(heads.values())
    if len(distinct) != 1 or not distinct.pop():
        raise VerdictInputError(f"receipts disagree on git_head: {heads}")
    return str(next(iter(heads.values())))


def render(sweep: Path, b3: Path, b37: Path | None) -> str:
    sweep_payload = load_receipt(sweep, label="sweep")
    b3_payload = load_receipt(b3, label="B3")
    payloads = [("sweep", sweep_payload), ("B3", b3_payload)]
    b37_payload: dict[str, Any] | None = None
    if b37 is not None:
        b37_payload = load_receipt(b37, label="B37")
        payloads.append(("B37", b37_payload))
    head = _same_head(*payloads)

    best = require(sweep_payload, "best_omp_num_threads", label="sweep")
    minima = require(sweep_payload, "per_omp_min_process_wall_seconds", label="sweep")
    lines = [
        "## Verdict — eight-term outer, transactional lane",
        "",
        f"Every figure below is read from a receipt at `{head}`; none is recalled.",
        "",
        "### Native denominator (swept, not chosen)",
        "",
        f"- Best OMP: **{best}**, from the frozen host set.",
        "- Per-thread minimum process wall:",
        "",
        "  | OMP | min wall (s) |",
        "  |---|---|",
    ]
    for value in sorted(minima, key=lambda item: int(item)):
        marker = " **(best)**" if int(value) == int(best) else ""
        lines.append(f"  | {value} | {float(minima[value]):.1f}{marker} |")

    lines += [
        "",
        "### B3",
        "",
        f"- Speedup (min over min): **{require(b3_payload, 'speedup_min_over_min', label='B3')}**",
        f"- Native min/median/max wall: {b3_payload.get('native_min_process_wall_seconds')}"
        f" / {b3_payload.get('native_median_process_wall_seconds')}"
        f" / {b3_payload.get('native_max_process_wall_seconds')} s",
        f"- JAX min/median/max wall: {b3_payload.get('jax_min_process_wall_seconds')}"
        f" / {b3_payload.get('jax_median_process_wall_seconds')}"
        f" / {b3_payload.get('jax_max_process_wall_seconds')} s",
        f"- Measured J parity gap (max): "
        f"{require(b3_payload, 'claim_boundary', 'measured_j_rel_gap_max', label='B3')}",
        f"- `nested_speed_claim`: "
        f"{require(b3_payload, 'claim_boundary', 'nested_speed_claim', label='B3')}",
        f"- Native OMP cited: "
        f"{require(b3_payload, 'claim_boundary', 'native_omp_num_threads', label='B3')}"
        f" (provenance "
        f"{require(b3_payload, 'claim_boundary', 'omp_provenance', label='B3')})",
    ]

    if b37_payload is None:
        lines += [
            "",
            "### B37",
            "",
            "**Not run.** No B37 receipt was supplied, so this section states"
            " nothing rather than carrying a placeholder a reader could mistake"
            " for a result.",
        ]
    else:
        lines += [
            "",
            "### B37",
            "",
            f"- Speedup (min over min): "
            f"**{require(b37_payload, 'speedup_min_over_min', label='B37')}**",
            f"- Measured J parity gap (max): "
            f"{require(b37_payload, 'claim_boundary', 'measured_j_rel_gap_max', label='B37')}",
            f"- Frozen J-parity band: "
            f"{require(b37_payload, 'claim_boundary', 'j_parity_rtol', label='B37')}",
            f"- `nested_speed_claim`: "
            f"{require(b37_payload, 'claim_boundary', 'nested_speed_claim', label='B37')}",
            f"- fail_closed_reason: {b37_payload.get('fail_closed_reason')}",
        ]

    lines += [
        "",
        "### Scope this verdict does NOT cover",
        "",
        "- These are **nested-lane** numbers. They are not F3's flat-675 result"
        "  and do not inherit its claim: handing the flat machinery a nested"
        "  problem ties native (2899 s vs 2876 s).",
        "- The inner lane each receipt ran is in its own `inner_policy` block."
        "  A receipt whose `trajectory_is_stock` is false is not comparable to"
        "  one where it is true.",
        "- Phase-4 tolerance-budget figures are **single-state** — one anchor,"
        "  one displacement, one host — and license no coarse tier.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", required=True, type=Path)
    parser.add_argument("--b3", required=True, type=Path)
    parser.add_argument("--b37", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        print(render(args.sweep, args.b3, args.b37))
    except VerdictInputError as error:
        print(f"verdict refused: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
