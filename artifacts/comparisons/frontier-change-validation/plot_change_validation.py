#!/usr/bin/env python3

import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    root = Path(__file__).resolve().parent
    observations = json.loads((root / "observations.json").read_text(encoding="utf-8"))[
        "observations"
    ]

    labels = [entry["subject_id"] for entry in observations]
    values = [1 if entry["status"] == "pass" else 0 for entry in observations]
    colors = ["#2e7d32" if value == 1 else "#c62828" for value in values]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(labels, values, color=colors)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Contract Match")
    ax.set_title("Frontier Change Validation")
    ax.set_yticks([0, 1], labels=["mismatch", "match"])
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(root / "change_validation_summary.png", dpi=180)


if __name__ == "__main__":
    main()
