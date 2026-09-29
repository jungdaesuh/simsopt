"""``simsoptpp.integral_BdotN`` gives the same bits for every OpenMP team.

The kernel computes its per-point terms in parallel and sums them serially in
index order (``src/simsoptpp/integral_BdotN.cpp``), so SquaredFlux does not
depend on how many threads evaluated it, nor on which call. An OpenMP
``reduction`` gave team- and call-dependent last bits, which moved native L-BFGS
endpoints and broke bitwise comparisons with one-thread references.

Each team size runs in its own child process because OpenMP reads
``OMP_NUM_THREADS`` when the extension is loaded, not per call. The child also
proves the team was applied: OpenMP starts its worker threads at the first
parallel region, so the first kernel call must add exactly ``team - 1`` threads
to the process. That fails, too, when the extension was built without OpenMP
(CMake links it only when it finds it), where every team size would trivially
agree.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TEAM_SIZES = ("1", "2", "4", "16")
_REPEATS = 20

# Child program: evaluate every definition, with and without a target, on a
# grid large enough to split across threads; repeat each call and print the
# exact bits of every result, one line per case.
_CHILD_PROGRAM = """
import sys
from pathlib import Path

repo_root, repeats = sys.argv[1], int(sys.argv[2])
sys.path.insert(0, repo_root)
from repo_bootstrap import bootstrap_local_simsopt

bootstrap_local_simsopt(Path(repo_root) / "src")

import os

import numpy as np
import simsoptpp as sopp

rng = np.random.default_rng(20260928)
nphi, ntheta = 96, 80
B = rng.standard_normal((nphi, ntheta, 3)) * rng.uniform(0.1, 10.0, (nphi, ntheta, 1))
normal = rng.standard_normal((nphi, ntheta, 3))
targets = {
    "no-target": np.zeros((0,)),
    "target": rng.standard_normal((nphi, ntheta)),
}
threads_before = len(os.listdir("/proc/self/task"))
sopp.integral_BdotN(B, targets["target"], normal, "quadratic flux")
print("started-threads", len(os.listdir("/proc/self/task")) - threads_before)
for definition in ("quadratic flux", "normalized", "local"):
    for target_name, target in targets.items():
        values = {
            float(sopp.integral_BdotN(B, target, normal, definition)).hex()
            for _ in range(repeats)
        }
        print(definition, target_name, ",".join(sorted(values)))
"""


def _bits_at_team_size(team_size: str) -> dict[tuple[str, str], str]:
    environment = dict(os.environ)
    environment["OMP_NUM_THREADS"] = team_size
    # Keep other thread pools out of the thread count the child reports.
    environment["OPENBLAS_NUM_THREADS"] = "1"
    completed = subprocess.run(
        (sys.executable, "-c", _CHILD_PROGRAM, str(_REPO_ROOT), str(_REPEATS)),
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    lines = completed.stdout.splitlines()
    started = lines[0].split()
    assert started[0] == "started-threads", completed.stdout
    assert int(started[1]) == int(team_size) - 1, (
        f"OMP_NUM_THREADS={team_size}: the first kernel call started "
        f"{started[1]} threads, so the OpenMP team was not applied (or the "
        "extension was built without OpenMP)"
    )
    bits = {}
    for line in lines[1:]:
        definition, target_name, values = line.rsplit(" ", 2)
        bits[(definition, target_name)] = values
    assert len(bits) == 6, completed.stdout
    return bits


def test_integral_bdotn_bits_do_not_depend_on_the_openmp_team() -> None:
    by_team = {team_size: _bits_at_team_size(team_size) for team_size in _TEAM_SIZES}

    for team_size, bits in by_team.items():
        for case, values in bits.items():
            assert "," not in values, (
                f"{case} varies between calls at OMP_NUM_THREADS={team_size}: {values}"
            )
    reference = by_team["1"]
    for team_size, bits in by_team.items():
        assert bits == reference, (
            f"OMP_NUM_THREADS={team_size} differs from one thread: "
            f"{ {case: (bits[case], reference[case]) for case in bits if bits[case] != reference[case]} }"
        )
