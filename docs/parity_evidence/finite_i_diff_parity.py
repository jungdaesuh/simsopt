"""Diff parity JSONs produced by parity_driver.py across pre-revert and HEAD.

Reports max abs/rel diffs on surface.x, iota, G; flags any nonmatching success
state and any fixture-hash mismatch.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path("/tmp/simsopt-prerevert-parity")
LANES = [
    ("ls_I0", "LS mode, I=0"),
    ("ls_Inz", "LS mode, I=mu0*5kA"),
    ("exact_I0", "exact mode, I=0"),
    ("exact_Inz", "exact mode, I=mu0*5kA"),
]


def _load(flavor: str, lane: str) -> dict:
    path = ROOT / f"parity_{flavor}_{lane}.json"
    with path.open() as f:
        return json.load(f)


def _rel(a, b):
    if a is None or b is None:
        return None
    denom = max(abs(a), abs(b), 1.0)
    return abs(a - b) / denom


def _abs(a, b):
    if a is None or b is None:
        return None
    return abs(a - b)


def diff_lane(lane: str, label: str) -> dict:
    pre = _load("prerevert", lane)
    head = _load("HEAD", lane)

    pre_x = np.asarray(pre["surface_x"], dtype=np.float64)
    head_x = np.asarray(head["surface_x"], dtype=np.float64)

    out = {
        "lane": lane,
        "label": label,
        "fixture_match": pre["fixture_hash"] == head["fixture_hash"],
        "fixture_hash_pre": pre["fixture_hash"],
        "fixture_hash_head": head["fixture_hash"],
        "success_pre": pre["success"],
        "success_head": head["success"],
        "iota_pre": pre["iota"],
        "iota_head": head["iota"],
        "G_pre": pre["G"],
        "G_head": head["G"],
        "surface_x_max_abs": float(np.max(np.abs(pre_x - head_x))),
        "surface_x_max_rel": (
            float(np.max(np.abs(pre_x - head_x) / np.maximum(np.maximum(np.abs(pre_x), np.abs(head_x)), 1.0)))
        ),
        "iota_abs": _abs(pre["iota"], head["iota"]),
        "iota_rel": _rel(pre["iota"], head["iota"]),
        "G_abs": _abs(pre["G"], head["G"]),
        "G_rel": _rel(pre["G"], head["G"]),
        "error_pre": pre.get("solve_error"),
        "error_head": head.get("solve_error"),
    }
    return out


def verdict(diff: dict) -> str:
    if not diff["fixture_match"]:
        return "FIXTURE_MISMATCH"
    if diff["success_pre"] != diff["success_head"]:
        return "SUCCESS_DIVERGED"
    if not diff["success_pre"]:
        # both failed — check whether they failed identically
        if diff["iota_pre"] is None or diff["iota_head"] is None:
            return "BOTH_FAIL_NAN_VS_NAN" if diff["iota_pre"] is None and diff["iota_head"] is None else "BOTH_FAIL_NAN_VS_NUMERIC"
        ix = diff["iota_rel"]
        gx = diff["G_rel"]
        if ix < 1e-12 and gx < 1e-12:
            return "BOTH_FAIL_IDENTICAL_TRAJECTORY"
        return "BOTH_FAIL_DIVERGED_TRAJECTORY"
    # both succeeded
    sx = diff["surface_x_max_rel"]
    ix = diff["iota_rel"]
    gx = diff["G_rel"]
    if sx < 1e-13 and ix < 1e-13 and gx < 1e-13:
        return "BIT_IDENTICAL"
    if sx < 1e-10 and ix < 1e-10 and gx < 1e-10:
        return "PARITY_AT_1e-10"
    if sx < 1e-6 and ix < 1e-6 and gx < 1e-6:
        return "PARITY_AT_1e-6"
    return "DRIFT_DETECTED"


def main():
    results = []
    for lane, label in LANES:
        d = diff_lane(lane, label)
        d["verdict"] = verdict(d)
        results.append(d)

    print(f"{'lane':<12} {'verdict':<32} {'iota_rel':<12} {'G_rel':<12} {'surf_x_max_rel':<14}")
    print("-" * 84)
    for d in results:
        ir = "N/A" if d["iota_rel"] is None else f"{d['iota_rel']:.3e}"
        gr = "N/A" if d["G_rel"] is None else f"{d['G_rel']:.3e}"
        sx = f"{d['surface_x_max_rel']:.3e}"
        print(f"{d['lane']:<12} {d['verdict']:<32} {ir:<12} {gr:<12} {sx:<14}")

    fixture_ok = all(d["fixture_match"] for d in results)
    print()
    print(f"fixture hash match across all lanes: {fixture_ok}")
    print(f"fixture hash: {results[0]['fixture_hash_pre']}")

    out_path = ROOT / "parity_evidence.json"
    with out_path.open("w") as f:
        json.dump(
            {
                "baseline_sha": "d8deb9e11 (= 459da8fab^)",
                "head_sha": "1002df7d6",
                "fixture_hash": results[0]["fixture_hash_pre"],
                "lanes": results,
            },
            f,
            indent=2,
        )
    print(f"\nevidence written to: {out_path}")


if __name__ == "__main__":
    main()
