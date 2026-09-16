#!/usr/bin/env python3
"""Render banana-coil geometry from a Stage-2 artifact, and optionally compare two.

Two artifact kinds are accepted:

  * a simsopt ``BiotSavart`` JSON (``biot_savart_opt.json``), which carries both the
    banana coils and the TF coils; they are separated by current magnitude.
  * a highres export directory holding ``coil1.csv`` .. ``coilN.csv`` whose first three
    columns are ``posx,posy,posz`` (the packaged o32 format). Banana coils only.

Usage::

    plot_coils.py ARTIFACT [ARTIFACT2] -o out.png

With one artifact it draws a 3D view, an R-Z projection of every coil, and coil 1
alone. With two it stacks them on shared axes and reports the deviation between them.

Deviation is measured as the nearest-point distance from each sample of coil 1 of the
first artifact to the polyline of coil 1 of the second. It is NOT the pointwise
distance between equal-index samples: the two artifacts need not share a curve
parameterization, and a phase offset inflates a pointwise metric by an order of
magnitude while the curves lie on top of each other.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from simsopt.field import BiotSavart

TF_CURRENT_FLOOR_A = 5e4
COLORS = ("#1f77b4", "#d62728")


def load_coils(artifact: Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Return ``(banana, tf)`` point arrays, each ``(n_points, 3)`` in metres."""
    if artifact.is_dir():
        files = sorted(artifact.glob("coil*.csv"), key=lambda p: int("".join(filter(str.isdigit, p.stem))))
        return [np.loadtxt(f, delimiter=",", skiprows=1, usecols=(0, 1, 2)) for f in files], []
    field = BiotSavart.from_file(str(artifact))
    banana = [c.curve.gamma() for c in field.coils if abs(c.current.get_value()) < TF_CURRENT_FLOOR_A]
    tf = [c.curve.gamma() for c in field.coils if abs(c.current.get_value()) >= TF_CURRENT_FLOOR_A]
    return banana, tf


def nearest_point_deviation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from every sample of ``a`` to the closest sample of ``b``, in metres."""
    return np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)).min(axis=1)


def coil_length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(np.vstack([points, points[:1]]), axis=0), axis=1).sum())


def _closed(points: np.ndarray) -> np.ndarray:
    return np.vstack([points, points[:1]])


def _draw_row(fig, row: int, rows: int, label: str, banana, tf, color: str, limits) -> None:
    r_lim, z_lim = limits
    axis_3d = fig.add_subplot(rows, 3, row * 3 + 1, projection="3d")
    for points in tf:
        axis_3d.plot(*_closed(points).T, color="0.85", lw=0.6)
    for points in banana:
        axis_3d.plot(*_closed(points).T, color=color, lw=1.6)
    axis_3d.set_title(label, fontsize=9)
    axis_3d.set_box_aspect((1, 1, 0.55))
    axis_3d.set_xlabel("x [m]"), axis_3d.set_ylabel("y [m]"), axis_3d.set_zlabel("z [m]")

    axis_rz = fig.add_subplot(rows, 3, row * 3 + 2)
    for points in banana:
        axis_rz.plot(np.hypot(points[:, 0], points[:, 1]), points[:, 2], color=color, lw=1.2)
    axis_rz.set_title(f"R-Z, all {len(banana)} coils", fontsize=9)

    first = banana[0]
    radius = np.hypot(first[:, 0], first[:, 1])
    axis_one = fig.add_subplot(rows, 3, row * 3 + 3)
    axis_one.plot(radius, first[:, 2], color=color, lw=2.2)
    axis_one.set_title(
        f"coil 1   R {radius.min():.3f}-{radius.max():.3f}   "
        f"Z {first[:, 2].min():+.3f}/{first[:, 2].max():+.3f}   L {coil_length(first):.4f} m",
        fontsize=8,
    )
    for axis in (axis_rz, axis_one):
        axis.set_xlabel("R [m]"), axis.set_ylabel("Z [m]")
        axis.set_aspect("equal"), axis.grid(alpha=0.3)
        axis.set_xlim(*r_lim), axis.set_ylim(*z_lim)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("artifacts", nargs="+", type=Path, help="biot_savart JSON, or a directory of coilN.csv")
    parser.add_argument("-o", "--output", type=Path, default=Path("coils.png"))
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--title", default=None)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    loaded = [load_coils(a) for a in args.artifacts]
    labels = args.labels or [a.name for a in args.artifacts]

    every = np.vstack([p for banana, _ in loaded for p in banana])
    radius = np.hypot(every[:, 0], every[:, 1])
    pad_r, pad_z = 0.02 * np.ptp(radius), 0.08 * np.ptp(every[:, 2])
    limits = ((radius.min() - pad_r, radius.max() + pad_r),
              (every[:, 2].min() - pad_z, every[:, 2].max() + pad_z))

    rows = len(loaded)
    figure = plt.figure(figsize=(15.5, 4.3 * rows))
    if args.title:
        figure.suptitle(args.title, fontsize=12)
    for row, ((banana, tf), label) in enumerate(zip(loaded, labels)):
        _draw_row(figure, row, rows, f"{label}  ({len(banana)} banana coils)", banana, tf,
                  COLORS[row % len(COLORS)], limits)
    figure.tight_layout()
    figure.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"wrote {args.output}")

    for (banana, _), label in zip(loaded, labels):
        first = banana[0]
        print(f"{label}: coil 1 length {coil_length(first):.4f} m, {len(first)} samples")
    if len(loaded) == 2:
        deviation = nearest_point_deviation(loaded[0][0][0], loaded[1][0][0])
        print(f"coil-1 geometric deviation: rms {deviation.mean() * 1e3:.3f} mm, "
              f"max {deviation.max() * 1e3:.3f} mm")


if __name__ == "__main__":
    main()
