"""Run one permanent-magnet native-cpu lane in a one-thread child process.

Every number an official capture carries was measured with ``OMP_NUM_THREADS=1``
on an independent official build. ``build_parity_lane_environment`` is the SSOT
that pins that for the ``native-cpu`` lane, and ``OMP_NUM_THREADS`` is read when
libgomp starts, so an in-process pin cannot undo the pytest process team
(``tests/conftest.py`` pins nothing). A comparison against an official number
therefore runs in a child.

The child builds the frozen input as well as running the lane. The construction
is not neutral: the QA coil pre-optimization is a ``scipy`` solve over an
OpenMP ``BiotSavart`` objective, and the MUSE/PM4Stell response matrices come
out of ``geo_setup_from_famus``; both feed the numbers being compared, so a
construction done in the pytest process would put a possibly multi-threaded
input under a one-thread official run. The bundle is written to ``bundle_root``
and the caller reads it back with ``read_input_bundle`` for the JAX lane, so
both lanes still consume identical bytes.

The same child also runs the conditioning probe of a bounded endpoint
(:func:`run_permanent_magnet_conditioning_probe`): the reference solve and every
perturbed solve are executed by ONE one-thread interpreter, so the difference
between them cannot be OpenMP reduction scatter.
"""

from __future__ import annotations

import pickle
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from examples.jax.parity.arbiter import LaneObservation
from parity_native_cpu import run_native_cpu_child

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Child source is a string so OpenMP is in the environment before the compiled
#: extension is imported. ``simsopt.util.coil_optimization_helper_functions``
#: resolves ``minimize`` as a module global, so replacing that attribute records
#: the parameter vector the coil pre-optimization provider actually receives,
#: and the same rebinding over ``execute_arbvec_case`` counts the GPMO solves
#: the lane runs -- the history and the endpoint below must come from ONE.
_CHILD_SOURCE = """\
import os
import pickle
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import simsopt.util.coil_optimization_helper_functions as coil_helper
from examples.jax.parity.cases import get_case
from examples.jax.parity.cases import native_permanent_magnet_muse as muse_case
from examples.jax.parity.cases import native_permanent_magnet_pm4stell as pm4stell_case
from examples.jax.parity.input_bundle import load_input_bundle

case_id = sys.argv[1]
scale = sys.argv[2]
bundle_root = Path(sys.argv[3])
out_path = Path(sys.argv[4])

provider_parameter_counts = []
library_minimize = coil_helper.minimize


def recording_minimize(fun, x0, **kwargs):
    provider_parameter_counts.append(int(np.asarray(x0).size))
    return library_minimize(fun, x0, **kwargs)


coil_helper.minimize = recording_minimize

gpmo_solve_lanes = []
gpmo_cases = {
    "native-permanent-magnet-muse": (muse_case, muse_case.GPMO_WORKFLOW_STAGES),
    "native-permanent-magnet-pm4stell": (
        pm4stell_case,
        pm4stell_case.WORKFLOW_STAGES,
    ),
}
for module, _stages in gpmo_cases.values():
    library_execute = module.execute_arbvec_case

    def counting_execute(lane, bundle, arrays, stages, _inner=library_execute):
        gpmo_solve_lanes.append(lane)
        return _inner(lane, bundle, arrays, stages)

    module.execute_arbvec_case = counting_execute

case = get_case(case_id)
bundle = case.create_input(bundle_root, scale)
_, arrays = load_input_bundle(bundle_root, bundle)
objective_history = None
if case_id in gpmo_cases:
    # ONE solve: the recorded history and the published endpoint are the same
    # run, not two runs of a solver the tests would have to assume is
    # deterministic. ``observe`` is the case's own derivation of its
    # observation from that solve, the second half of its ``execute``.
    module, stages = gpmo_cases[case_id]
    result = module.execute_arbvec_case("native-cpu", bundle, arrays, stages)
    objective_history = np.asarray(result.objective_history, dtype=np.float64)
    observation = module.observe("native-cpu", bundle, arrays, result)
else:
    observation = case.execute("native-cpu", bundle, arrays)
fields_payload = {field.name: getattr(observation, field.name) for field in fields(observation)}
fields_payload["values"] = dict(observation.values)
fields_payload["applicability"] = dict(observation.applicability)
out_path.write_bytes(
    pickle.dumps(
        {
            "observation": fields_payload,
            "configuration": dict(bundle.configuration),
            "objective_history": objective_history,
            "provider_parameter_counts": tuple(provider_parameter_counts),
            "native_solve_count": sum(
                1 for lane in gpmo_solve_lanes if lane == "native-cpu"
            ),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        }
    )
)
"""

#: The conditioning probe. The perturbation is the campaign's pre-registered
#: one-ulp rule (``A/d1-diagnostic/NOTES.md``, rule v2): every entry of the
#: named input array is moved to its neighbouring float64,
#: ``np.nextafter(x, s * inf)`` with ``s in {-1, +1}^n`` drawn from
#: ``RandomState(20260920 + k)`` for k = 1..draws. The child returns the
#: reference and every draw so the caller can compare per observable.
_CONDITIONING_CHILD_SOURCE = """\
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from examples.jax.parity.cases import get_case
from examples.jax.parity.input_bundle import load_input_bundle

case_id = sys.argv[1]
scale = sys.argv[2]
bundle_root = Path(sys.argv[3])
out_path = Path(sys.argv[4])
perturbed_array = sys.argv[5]
draws = int(sys.argv[6])
seed_base = int(sys.argv[7])
observables = tuple(sys.argv[8:])

case = get_case(case_id)
bundle = case.create_input(bundle_root, scale)
_, arrays = load_input_bundle(bundle_root, bundle)


def measure(values):
    return {
        "values": {name: float(np.asarray(values[name])) for name in observables},
        "nonzero_mask": np.asarray(values["final:nonzero_mask"], dtype=bool),
    }


reference = measure(case.execute("native-cpu", bundle, arrays).values)
source = arrays[perturbed_array]
records = []
for k in range(1, draws + 1):
    signs = np.where(
        np.random.RandomState(seed_base + k).randint(0, 2, source.shape) == 0, -1.0, 1.0
    )
    perturbed = {name: np.array(value, copy=True) for name, value in arrays.items()}
    perturbed[perturbed_array] = np.nextafter(source, signs * np.inf)
    moved = measure(case.execute("native-cpu", bundle, perturbed).values)
    moved["seed"] = seed_base + k
    # One ulp, per entry, in the drawn direction: the count of entries whose
    # move is NOT exactly one ulp is reported so the probe cannot silently
    # become a large-perturbation test.
    moved["entries_moved_more_than_one_ulp"] = int(
        np.count_nonzero(
            np.abs(perturbed[perturbed_array] - source) > np.abs(np.spacing(source))
        )
    )
    records.append(moved)

out_path.write_bytes(
    pickle.dumps(
        {
            "reference": reference,
            "draws": records,
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        }
    )
)
"""


@dataclass(frozen=True)
class NativeChildResult:
    """What the one-thread child measured."""

    observation: LaneObservation
    #: The frozen configuration the child built, as the bundle carries it.
    configuration: dict[str, object]
    #: ``R2`` per recorded row for the arbitrary-vector GPMO cases, else ``None``.
    objective_history: np.ndarray | None
    #: Size of every parameter vector a ``scipy`` provider received during the
    #: construction, in call order.
    provider_parameter_counts: tuple[int, ...]
    #: How many native GPMO solves the child ran. The history and the endpoint
    #: are the same run only if this is one.
    native_solve_count: int
    #: The child's own ``OMP_NUM_THREADS``; the tests assert it is ``"1"``.
    omp_num_threads: str | None


@dataclass(frozen=True)
class ConditioningDraw:
    """One perturbed solve of the conditioning probe."""

    seed: int
    values: Mapping[str, float]
    nonzero_mask: np.ndarray
    #: Entries of the perturbed array that moved by more than one ulp.
    entries_moved_more_than_one_ulp: int


@dataclass(frozen=True)
class ConditioningProbeResult:
    """One lane's endpoint under the pre-registered one-ulp input perturbations."""

    reference: Mapping[str, float]
    reference_nonzero_mask: np.ndarray
    draws: tuple[ConditioningDraw, ...]
    omp_num_threads: str | None

    def relative_drifts(self, observable: str) -> tuple[float, ...]:
        """|moved - reference| / |reference| of ``observable``, one per draw."""
        reference = self.reference[observable]
        return tuple(
            abs(draw.values[observable] - reference) / abs(reference)
            for draw in self.draws
        )


def run_permanent_magnet_native_lane(
    case_id: str,
    scale: str,
    bundle_root: Path,
    payload_path: Path,
) -> NativeChildResult:
    """Build the input and run the native lane of ``case_id`` at one thread."""
    completed = run_native_cpu_child(
        _CHILD_SOURCE,
        case_id,
        scale,
        str(bundle_root),
        str(payload_path),
        repo_root=REPO_ROOT,
    )
    assert completed.returncode == 0, completed.stderr
    payload = pickle.loads(payload_path.read_bytes())
    return NativeChildResult(
        observation=LaneObservation(**payload["observation"]),
        configuration=payload["configuration"],
        objective_history=payload["objective_history"],
        provider_parameter_counts=payload["provider_parameter_counts"],
        native_solve_count=payload["native_solve_count"],
        omp_num_threads=payload["omp_num_threads"],
    )


def run_permanent_magnet_conditioning_probe(
    case_id: str,
    scale: str,
    bundle_root: Path,
    payload_path: Path,
    *,
    perturbed_array: str,
    observables: Sequence[str],
    draws: int,
    seed_base: int,
) -> ConditioningProbeResult:
    """Solve once unperturbed and ``draws`` times one ulp away, in ONE child."""
    completed = run_native_cpu_child(
        _CONDITIONING_CHILD_SOURCE,
        case_id,
        scale,
        str(bundle_root),
        str(payload_path),
        perturbed_array,
        str(draws),
        str(seed_base),
        *observables,
        repo_root=REPO_ROOT,
    )
    assert completed.returncode == 0, completed.stderr
    payload = pickle.loads(payload_path.read_bytes())
    return ConditioningProbeResult(
        reference=payload["reference"]["values"],
        reference_nonzero_mask=payload["reference"]["nonzero_mask"],
        draws=tuple(
            ConditioningDraw(
                seed=int(record["seed"]),
                values=record["values"],
                nonzero_mask=record["nonzero_mask"],
                entries_moved_more_than_one_ulp=int(
                    record["entries_moved_more_than_one_ulp"]
                ),
            )
            for record in payload["draws"]
        ),
        omp_num_threads=payload["omp_num_threads"],
    )
