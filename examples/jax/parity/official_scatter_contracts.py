"""Contracts derived from upstream's own end-state scatter at the parity harness's scales.

Both derivations read one tracked upstream scatter record (``official_reference/9e027eac3/scatter/``) and nothing
else; no branch value, native or JAX, enters either. Each uses only the pre-registered draws ``k = 0..8`` (the
nine), fixed before any sample was drawn:

* ``upstream_scatter_quality_band`` -- the rule-v2 band ``max(S) * (1 + (max(S) - min(S)) / min(S))`` over the nine
  upstream end values of one observable, for a workflow whose end VALUE is path dependent;
* ``upstream_end_states`` -- upstream's end states of the nine, for a workflow whose end STATE (which solution it
  lands on) is not determined by its input: every draw whose named upstream stages all succeeded enters the set,
  and the arbiter admits a lane whose end state matches at least one of them under the case's own route comparators.

Both are engineering acceptances against upstream's own scatter, never an equivalence proof; the arbiter can
therefore only return ``quality-band`` under them. Neither may be declared without a tracked same-state test proving
the lanes compute upstream's function (and gradient, where the workflow uses one) at upstream's recorded states; the
``same_state_proof`` argument names it and is carried into the derivation.
"""

from __future__ import annotations

from typing import Final

from examples.jax.parity.contracts import (
    QualityBand,
    UpstreamEndState,
    UpstreamEndStates,
)
from examples.jax.parity.official_quality_bands import upstream_scatter_ceiling
from examples.jax.parity.official_reference import (
    OfficialUpstreamScatter,
    UpstreamScatterRun,
    load_upstream_scatter,
)
from simsopt_jax.examples import ExecutionScale

#: The pre-registered draws every contract is derived from (k = 0 unperturbed, k = 1..8 one ulp off the start).
PRE_REGISTERED_DRAWS: Final[tuple[int, ...]] = tuple(range(9))


def pre_registered_runs(
    scatter: OfficialUpstreamScatter,
) -> tuple[UpstreamScatterRun, ...]:
    """The record's nine pre-registered runs, in k order; any other run is never part of a contract."""
    runs = tuple(run for run in scatter.runs if run.k in PRE_REGISTERED_DRAWS)
    if tuple(run.k for run in runs) != PRE_REGISTERED_DRAWS:
        raise ValueError(
            f"{scatter.case_id} ({scatter.scale}): the record lacks pre-registered draws "
            f"{sorted(set(PRE_REGISTERED_DRAWS) - {run.k for run in runs})}"
        )
    return runs


def _provenance(scatter: OfficialUpstreamScatter) -> str:
    runner = (
        "the official script verbatim"
        if scatter.runner.kind == "verbatim"
        else "the official script with only its scale lines changed (record: runner.body_diff)"
    )
    return (
        f"upstream {scatter.upstream_commit[:9]} {scatter.official_script}, {runner}, one thread, "
        f"one-ulp start protocol, pre-registered draws k = 0..8"
    )


def upstream_scatter_quality_band(
    case_id: str,
    scale: ExecutionScale,
    observable: str,
    *,
    same_state_proof: str,
    disclosure: str = "",
) -> QualityBand:
    """The rule-v2 band of ``observable`` over upstream's nine end values at ``scale``."""
    scatter = load_upstream_scatter(case_id, scale)
    samples = tuple(
        float(run.value(observable)) for run in pre_registered_runs(scatter)
    )
    return QualityBand(
        observable=observable,
        max_value=upstream_scatter_ceiling(samples),
        derivation=(
            f"upstream scatter at {scale}: {_provenance(scatter)}; end values of {observable} "
            f"(upstream key {scatter.capture_keys[observable]}) min {min(samples)!r}, max {max(samples)!r}; "
            "ceiling = max(S) * (1 + (max(S) - min(S)) / min(S)); gross-failure guard; "
            f"same-state proof: {same_state_proof}"
            + (f"; {disclosure}" if disclosure else "")
        ),
        scale=scale,
    )


def upstream_end_states(
    case_id: str,
    scale: ExecutionScale,
    observables: tuple[str, ...],
    *,
    same_state_proof: str,
    disclosure: str = "",
) -> UpstreamEndStates:
    """Upstream's nine end states at ``scale`` whose named stages all succeeded, keyed by lane observable."""
    scatter = load_upstream_scatter(case_id, scale)
    runs = pre_registered_runs(scatter)
    states = tuple(
        UpstreamEndState(
            k=run.k,
            values={observable: run.value(observable) for observable in observables},
        )
        for run in runs
        if run.workflow_success
    )
    failed = [run.k for run in runs if not run.workflow_success]
    return UpstreamEndStates(
        case_id=case_id,
        scale=scale,
        observables=observables,
        states=states,
        derivation=(
            f"upstream end states at {scale}: {_provenance(scatter)}; the {len(states)} draws whose upstream "
            f"stages all succeeded (failed: {failed or 'none'}); a lane passes when its end state matches one of "
            "them under the case's own route comparator and tolerance for every judged key; "
            f"same-state proof: {same_state_proof}"
            + (f"; {disclosure}" if disclosure else "")
        ),
    )
