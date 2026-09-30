"""Static registry of typed native/JAX parity cases."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
from examples.jax.parity.arbiter import LaneObservation
from examples.jax.parity.cases.native_boozer import (
    END_STATE_OBSERVABLES as NATIVE_BOOZER_END_STATE_OBSERVABLES,
)
from examples.jax.parity.cases.native_boozer import (
    REPLAY_EXACT_OBSERVABLES as NATIVE_BOOZER_REPLAY_EXACT_OBSERVABLES,
)
from examples.jax.parity.cases.native_boozer import (
    REPLAY_DERIVED_OBSERVABLES as NATIVE_BOOZER_REPLAY_DERIVED_OBSERVABLES,
)
from examples.jax.parity.cases.native_boozer import (
    REPLAY_SOLUTION_OBSERVABLES as NATIVE_BOOZER_REPLAY_SOLUTION_OBSERVABLES,
)
from examples.jax.parity.cases.native_boozer import (
    create_input as create_native_boozer_input,
)
from examples.jax.parity.cases.native_boozer import execute as execute_native_boozer
from examples.jax.parity.cases.native_boozerqa import (
    create_input as create_native_boozerqa_input,
)
from examples.jax.parity.cases.native_boozerqa import (
    execute as execute_native_boozerqa,
)
from examples.jax.parity.cases.native_coil_forces import (
    create_input as create_native_coil_forces_input,
)
from examples.jax.parity.cases.native_coil_forces import (
    execute as execute_native_coil_forces,
)
from examples.jax.parity.cases.native_just_a_quadratic import (
    create_input as create_native_just_a_quadratic_input,
)
from examples.jax.parity.cases.native_just_a_quadratic import (
    execute as execute_native_just_a_quadratic,
)
from examples.jax.parity.cases.native_minimize_curve_length import (
    create_input as create_native_minimize_curve_length_input,
)
from examples.jax.parity.cases.native_minimize_curve_length import (
    execute as execute_native_minimize_curve_length,
)
from examples.jax.parity.cases.native_permanent_magnet_muse import (
    create_input as create_native_permanent_magnet_muse_input,
)
from examples.jax.parity.cases.native_permanent_magnet_muse import (
    execute as execute_native_permanent_magnet_muse,
)
from examples.jax.parity.cases.native_permanent_magnet_pm4stell import (
    create_input as create_native_permanent_magnet_pm4stell_input,
)
from examples.jax.parity.cases.native_permanent_magnet_pm4stell import (
    execute as execute_native_permanent_magnet_pm4stell,
)
from examples.jax.parity.cases.native_permanent_magnet_qa import (
    create_input as create_native_permanent_magnet_qa_input,
)
from examples.jax.parity.cases.native_permanent_magnet_qa import (
    execute as execute_native_permanent_magnet_qa,
)
from examples.jax.parity.cases.native_permanent_magnet_simple import (
    create_input as create_native_permanent_magnet_simple_input,
)
from examples.jax.parity.cases.native_permanent_magnet_simple import (
    execute as execute_native_permanent_magnet_simple,
)
from examples.jax.parity.cases.native_qfm import (
    create_input as create_native_qfm_input,
)
from examples.jax.parity.cases.native_qfm import execute as execute_native_qfm
from examples.jax.parity.cases.native_stage_two_optimization import (
    create_input as create_native_stage_two_optimization_input,
)
from examples.jax.parity.cases.native_stage_two_optimization import (
    execute as execute_native_stage_two_optimization,
)
from examples.jax.parity.cases.native_stage_two_optimization_finitebuild import (
    create_input as create_native_stage_two_finitebuild_input,
)
from examples.jax.parity.cases.native_stage_two_optimization_finitebuild import (
    execute as execute_native_stage_two_finitebuild,
)
from examples.jax.parity.cases.native_stage_two_optimization_minimal import (
    create_input as create_native_stage_two_optimization_minimal_input,
)
from examples.jax.parity.cases.native_stage_two_optimization_minimal import (
    execute as execute_native_stage_two_optimization_minimal,
)
from examples.jax.parity.cases.native_stage_two_optimization_planar_coils import (
    create_input as create_native_stage_two_optimization_planar_coils_input,
)
from examples.jax.parity.cases.native_stage_two_optimization_planar_coils import (
    execute as execute_native_stage_two_optimization_planar_coils,
)
from examples.jax.parity.cases.native_stage_two_optimization_stochastic import (
    create_input as create_native_stage_two_optimization_stochastic_input,
)
from examples.jax.parity.cases.native_stage_two_optimization_stochastic import (
    execute as execute_native_stage_two_optimization_stochastic,
)
from examples.jax.parity.cases.native_strain_optimization import (
    create_input as create_native_strain_optimization_input,
)
from examples.jax.parity.cases.native_strain_optimization import (
    execute as execute_native_strain_optimization,
)
from examples.jax.parity.cases.native_surf_vol_area import (
    create_input as create_native_surf_vol_area_input,
)
from examples.jax.parity.cases.native_surf_vol_area import (
    execute as execute_native_surf_vol_area,
)
from examples.jax.parity.cases.native_tracing_fieldlines_ncsx import (
    create_input as create_native_tracing_fieldlines_ncsx_input,
)
from examples.jax.parity.cases.native_tracing_fieldlines_ncsx import (
    execute as execute_native_tracing_fieldlines_ncsx,
)
from examples.jax.parity.cases.native_tracing_fieldlines_qa import (
    create_input as create_native_tracing_fieldlines_qa_input,
)
from examples.jax.parity.cases.native_tracing_fieldlines_qa import (
    execute as execute_native_tracing_fieldlines_qa,
)
from examples.jax.parity.cases.native_tracing_particle import (
    create_input as create_native_tracing_particle_input,
)
from examples.jax.parity.cases.native_tracing_particle import (
    execute as execute_native_tracing_particle,
)
from examples.jax.parity.cases.native_wireframe_gsco_modular import (
    create_input as create_native_wireframe_gsco_modular_input,
)
from examples.jax.parity.cases.native_wireframe_gsco_modular import (
    execute as execute_native_wireframe_gsco_modular,
)
from examples.jax.parity.cases.native_wireframe_gsco_multistep import (
    create_input as create_native_wireframe_gsco_multistep_input,
)
from examples.jax.parity.cases.native_wireframe_gsco_multistep import (
    execute as execute_native_wireframe_gsco_multistep,
)
from examples.jax.parity.cases.native_wireframe_gsco_sector_saddle import (
    create_input as create_native_wireframe_gsco_sector_saddle_input,
)
from examples.jax.parity.cases.native_wireframe_gsco_sector_saddle import (
    execute as execute_native_wireframe_gsco_sector_saddle,
)
from examples.jax.parity.cases.native_wireframe_rcls_basic import (
    create_input as create_native_wireframe_rcls_basic_input,
)
from examples.jax.parity.cases.native_wireframe_rcls_basic import (
    execute as execute_native_wireframe_rcls_basic,
)
from examples.jax.parity.cases.native_wireframe_rcls_with_ports import (
    create_input as create_native_wireframe_rcls_with_ports_input,
)
from examples.jax.parity.cases.native_wireframe_rcls_with_ports import (
    execute as execute_native_wireframe_rcls_with_ports,
)
from examples.jax.parity.contracts import (
    AdmittedTerminalOutcome,
    QualityBand,
    StageWiseContract,
    UpstreamEndStates,
)
from examples.jax.parity.input_bundle import InputBundle
from examples.jax.parity.measurement import MeasurementExecution
from examples.jax.parity.official_quality_bands import official_quality_band
from examples.jax.parity.official_reference import load_upstream_scatter
from examples.jax.parity.official_scatter_contracts import upstream_end_states
from examples.jax.parity.runtime import ParityLane
from examples.jax.parity.work_budget import WorkBudgetContract
from simsopt_jax.examples import ExecutionScale
from simsopt_jax.examples.solver_terminal_status import GSCO_REDUCED_BUDGET_SCALES


@dataclass(frozen=True)
class CaseDefinition:
    case_id: str
    create_input: Callable[[Path, ExecutionScale], InputBundle]
    execute: Callable[[ParityLane, InputBundle, dict[str, np.ndarray]], LaneObservation]
    measurement_execute: (
        Callable[
            [
                ParityLane,
                InputBundle,
                dict[str, np.ndarray],
                MeasurementExecution,
            ],
            LaneObservation,
        ]
        | None
    ) = None
    quality_bands: tuple[QualityBand, ...] = ()
    """Endpoint quality floors (2026-08-15 rule 3), at most one per scale.

    Each band is certifiable at its own ``scale`` only. No band -- the default
    for every case -- keeps the certification-gate behaviour unchanged there.
    """
    work_budget_contract: WorkBudgetContract | None = None
    native_default_admitted_terminal_outcomes: tuple[AdmittedTerminalOutcome, ...] = ()
    """Provider failures admitted for band judgment at ``native_default`` only.

    Each entry is an explicitly authorized case-specific composite ``raw_status`` for one lane, and
    upstream's own official script produced exactly that composite outcome under the one-ulp start
    protocol (the evidence text names the draws). The lane keeps its raw and normalized status and its
    ``success=False``, every finite and physical check stays in force, and the verdict can only be
    ``quality-band`` (an engineering endpoint acceptance, not an equivalence proof).
    """
    upstream_end_states: tuple[UpstreamEndStates, ...] = ()
    """Upstream's own end-state sets, at most one per scale (the 2026-09-29 redesign, C3).

    At a scale where upstream's own one-ulp draws land on several end states, each lane must match
    one of them on every judged key; this is an engineering acceptance against upstream's scatter,
    never an equivalence proof, and it admits no budget or failure outcome. At a scale that also
    declares a stage-wise contract the set is matched and recorded, informational.
    """
    stage_wise_contracts: tuple[StageWiseContract, ...] = ()
    """Stage-wise contracts, at most one per scale (PLAN.md amendment 5).

    At such a scale the stages are judged from shared states and the chained end point is
    informational; the verdict is ``quality-band`` at most.
    """

    def quality_band(self, scale: ExecutionScale) -> QualityBand | None:
        """The band this case declares at ``scale``, if any."""
        return next((band for band in self.quality_bands if band.scale == scale), None)

    def end_states(self, scale: ExecutionScale) -> UpstreamEndStates | None:
        """The upstream end-state set this case declares at ``scale``, if any."""
        return next(
            (states for states in self.upstream_end_states if states.scale == scale),
            None,
        )

    def stage_wise(self, scale: ExecutionScale) -> StageWiseContract | None:
        """The stage-wise contract this case declares at ``scale``, if any."""
        return next(
            (
                contract
                for contract in self.stage_wise_contracts
                if contract.scale == scale
            ),
            None,
        )

    def __post_init__(self) -> None:
        band_scales = [band.scale for band in self.quality_bands]
        if len(set(band_scales)) != len(band_scales):
            raise ValueError("case declares more than one quality band at one scale")
        end_state_scales = [states.scale for states in self.upstream_end_states]
        if len(set(end_state_scales)) != len(end_state_scales):
            raise ValueError(
                "case declares more than one upstream end-state set at one scale"
            )
        foreign = sorted(
            states.case_id
            for states in self.upstream_end_states
            if states.case_id != self.case_id
        )
        if foreign:
            raise ValueError(
                f"upstream end-state set belongs to another case: {foreign} != {self.case_id!r}"
            )
        if set(band_scales) & set(end_state_scales):
            raise ValueError(
                "case cannot combine a quality band and an upstream end-state set at one scale"
            )
        stage_wise_scales = [contract.scale for contract in self.stage_wise_contracts]
        if len(set(stage_wise_scales)) != len(stage_wise_scales):
            raise ValueError(
                "case declares more than one stage-wise contract at one scale"
            )
        foreign_stage_wise = sorted(
            contract.case_id
            for contract in self.stage_wise_contracts
            if contract.case_id != self.case_id
        )
        if foreign_stage_wise:
            raise ValueError(
                "stage-wise contract belongs to another case: "
                f"{foreign_stage_wise} != {self.case_id!r}"
            )
        if set(band_scales) & set(stage_wise_scales):
            raise ValueError(
                "case cannot combine a quality band and a stage-wise contract at one scale"
            )
        if self.work_budget_contract is not None and set(band_scales) & set(
            self.work_budget_contract.scales
        ):
            raise ValueError(
                "case cannot combine a quality band and work budget at one scale"
            )
        # An end-state set admits no budget exit; a work budget beside it would.
        if self.work_budget_contract is not None and set(end_state_scales) & set(
            self.work_budget_contract.scales
        ):
            raise ValueError(
                "case cannot combine an upstream end-state set and work budget at one scale"
            )
        if (
            self.native_default_admitted_terminal_outcomes
            and self.quality_band("native_default") is None
        ):
            raise ValueError(
                "admitted terminal outcomes require a native_default quality band"
            )


_FIXED_BUDGET_SCALES: tuple[ExecutionScale, ...] = ("bounded", "native_default")

#: Upstream's own composite terminal outcomes of ``coil_forces.py`` other than the budget pair
#: ``"1,1"`` (official 9e027eac3 build, one thread, one-ulp start protocol with signs from
#: ``RandomState(20260920 + k)``, k = 0..40), each with the draws that produced it. ``"2,1"`` is in
#: the tracked nine (k = 5); ``"1,2"`` and ``"2,2"`` come from the pre-registered extension
#: k = 9..40. Stage two ended ABNORMAL (SciPy status 2) on 5 of the 41 starts, every time at
#: stage-two nit 0: stage two restarts L-BFGS-B cold at stage one's end point, where the stage-two
#: length penalty is exactly zero, and the first trial step overshoots until ``maxls`` runs out.
COIL_FORCES_UPSTREAM_TERMINAL_OUTCOMES: Mapping[str, tuple[int, ...]] = (
    MappingProxyType({"2,1": (5,), "1,2": (20, 23, 35), "2,2": (19, 34)})
)
_COIL_FORCES_UPSTREAM_DRAWS = 41


_CASES = {
    "native-boozer": CaseDefinition(
        case_id="native-boozer",
        create_input=create_native_boozer_input,
        execute=execute_native_boozer,
        # Which Boozer surface the workflow lands on is not a function of its input: the first
        # stage stops at its 300-iteration L-BFGS cap, unconverged, and upstream's own official
        # script reaches 2 surfaces from nine one-ulp starts at native_default and 5 at the bounded
        # scale (tracked upstream scatter records). The workflow is therefore judged stage by stage
        # (PLAN.md amendment 5, B1): the official Newton stages replayed by every lane from the same
        # starts decide, and the chained end state is informational -- each lane's match against
        # upstream's own end states below is recorded, and no longer decides.
        stage_wise_contracts=tuple(
            StageWiseContract(
                case_id="native-boozer",
                scale=scale,
                informational_observables=(
                    *NATIVE_BOOZER_END_STATE_OBSERVABLES,
                    *NATIVE_BOOZER_REPLAY_DERIVED_OBSERVABLES,
                ),
                deciding_observables=(
                    *NATIVE_BOOZER_REPLAY_EXACT_OBSERVABLES,
                    *NATIVE_BOOZER_REPLAY_SOLUTION_OBSERVABLES,
                ),
                same_state_tests=(
                    "tests/integration/test_jax_mirror_boozer_official_end_states.py",
                    "tests/integration/test_jax_mirror_boozer_first_stage_same_state.py",
                ),
                derivation=(
                    "PLAN.md amendment 5 part 1, B1 (registered 2026-09-29 16:24 EDT, before any "
                    "formal run): from each start -- the native lane's own first-stage end, which "
                    "each JAX lane reruns in-process under its one-thread policy, then upstream's "
                    f"pre-registered first-stage ends k = 0..8 at {scale} (9e027eac3, one thread, "
                    "one-ulp protocol; tracked scatter record keys first:surface_dofs, first:iota, "
                    "first:G) -- every lane runs the official area and flux Newton stages; the starts "
                    "and success flags are compared exactly, every replayed solve must meet upstream's "
                    "success rule norm(J^T r) <= tol, and the two lanes' solved states must lie "
                    "within (norm(b_native) + norm(b_jax)) / min(lambda_min(J^T J)) plus "
                    "max(norm(dx*/dt)) times the two lanes' label-target difference of each other "
                    "(amendments 6 and 7, B1' and B1'', set post hoc by the user after the 1e-11 rule "
                    "and then the stopping-radius rule failed at the bounded k8 start; labels and "
                    "the flux target informational). The chained end "
                    "state is informational: the first stage's capped, "
                    "path-dependent L-BFGS end point decides which surface the chain reaches. "
                    "The first stage itself is judged at identical x (x0 and upstream's nine "
                    "first-stage ends) against a derived first-order rounding bound (amendment 5 "
                    "part 2, B2; an engineering error envelope on the GPU in its rsqrt constant)"
                ),
            )
            for scale in ("bounded", "native_default")
        ),
        upstream_end_states=tuple(
            upstream_end_states(
                load_upstream_scatter("native-boozer", scale),
                NATIVE_BOOZER_END_STATE_OBSERVABLES,
                same_state_proof=(
                    "tests/integration/test_jax_mirror_boozer_official_end_states.py (the JAX "
                    "penalty residual and Jacobian equal native's at upstream's recorded area and "
                    "flux end states of every branch representative this set admits) and the "
                    "initial-state comparison in tests/integration/test_jax_mirror_boozer_parity.py"
                ),
                disclosure=disclosure,
            )
            for scale, disclosure in (
                (
                    "bounded",
                    (
                        "at this scale the JAX lanes run a 60-iteration first stage where "
                        "upstream and the native lane run 300; the set holds the end states "
                        "of upstream's 300-iteration workflow"
                    ),
                ),
                ("native_default", ""),
            )
        ),
    ),
    "native-boozerqa": CaseDefinition(
        case_id="native-boozerqa",
        create_input=create_native_boozerqa_input,
        execute=execute_native_boozerqa,
        # Declared POST HOC on the user's decision C18 (2026-09-20 23:48 EDT), after the
        # lane-against-lane end-point routes had failed at native_default with the old and
        # the new inner route alike: upstream never converges here (BFGS at its 1000-
        # iteration cap), and its own end objective scatters by 12 % under one-ulp start
        # perturbations (tracked sensitivity record). Same rule v2 as the three pre-
        # registered band cases; the campaign record says so where the values were seen.
        quality_bands=(official_quality_band("native-boozerqa"),),
        work_budget_contract=WorkBudgetContract(
            # At native_default the band admits the cap (a band and a work budget cannot
            # both cover that scale); the reduced scales keep the fixed-budget contract.
            scales=("bounded",),
            derivation="Upstream BoozerQA fixes outer BFGS work by MAXITER; an accepted endpoint may stop at that cap.",
        ),
    ),
    "native-coil-forces": CaseDefinition(
        case_id="native-coil-forces",
        create_input=create_native_coil_forces_input,
        execute=execute_native_coil_forces,
        # Both stages are path dependent: upstream's own official script ends a stage ABNORMAL on 6 of
        # 41 one-ulp starts (COIL_FORCES_UPSTREAM_TERMINAL_OUTCOMES), so each such composite outcome is
        # admitted on every lane (the 2026-09-29 ruling: judge by upstream's own scatter; C14 admitted
        # "2,2" on the JAX GPU lane only). Same rule v2 band as the other band cases; the band is an
        # engineering endpoint acceptance, not an equivalence proof.
        quality_bands=(official_quality_band("native-coil-forces"),),
        work_budget_contract=WorkBudgetContract(
            # At native_default the band admits the cap (a band and a work budget cannot both cover
            # that scale); the reduced scale keeps the fixed-budget contract.
            scales=("bounded",),
            derivation="Upstream coil-forces stages use fixed L-BFGS-B iteration caps; accepted endpoints may exhaust the stage budgets.",
        ),
        native_default_admitted_terminal_outcomes=tuple(
            AdmittedTerminalOutcome(
                case_id="native-coil-forces",
                lane=lane,
                raw_status=raw_status,
                normalized_status="failed",
                upstream_evidence=(
                    f"official coil_forces.py at 9e027eac3, one thread, one-ulp start protocol: upstream "
                    f"ended with composite status {raw_status} on k = {', '.join(map(str, draws))} of "
                    f"k = 0..{_COIL_FORCES_UPSTREAM_DRAWS - 1}"
                ),
            )
            for lane in ("native-cpu", "jax-cpu", "jax-gpu")
            for raw_status, draws in COIL_FORCES_UPSTREAM_TERMINAL_OUTCOMES.items()
        ),
    ),
    "native-just-a-quadratic": CaseDefinition(
        case_id="native-just-a-quadratic",
        create_input=create_native_just_a_quadratic_input,
        execute=execute_native_just_a_quadratic,
    ),
    "native-minimize-curve-length": CaseDefinition(
        case_id="native-minimize-curve-length",
        create_input=create_native_minimize_curve_length_input,
        execute=execute_native_minimize_curve_length,
    ),
    "native-permanent-magnet-simple": CaseDefinition(
        case_id="native-permanent-magnet-simple",
        create_input=create_native_permanent_magnet_simple_input,
        execute=execute_native_permanent_magnet_simple,
    ),
    "native-permanent-magnet-muse": CaseDefinition(
        case_id="native-permanent-magnet-muse",
        create_input=create_native_permanent_magnet_muse_input,
        execute=execute_native_permanent_magnet_muse,
    ),
    "native-permanent-magnet-qa": CaseDefinition(
        case_id="native-permanent-magnet-qa",
        create_input=create_native_permanent_magnet_qa_input,
        execute=execute_native_permanent_magnet_qa,
    ),
    "native-permanent-magnet-pm4stell": CaseDefinition(
        case_id="native-permanent-magnet-pm4stell",
        create_input=create_native_permanent_magnet_pm4stell_input,
        execute=execute_native_permanent_magnet_pm4stell,
    ),
    "native-qfm": CaseDefinition(
        case_id="native-qfm",
        create_input=create_native_qfm_input,
        execute=execute_native_qfm,
        quality_bands=(official_quality_band("native-qfm"),),
    ),
    "native-surf-vol-area": CaseDefinition(
        case_id="native-surf-vol-area",
        create_input=create_native_surf_vol_area_input,
        execute=execute_native_surf_vol_area,
    ),
    "native-tracing-fieldlines-ncsx": CaseDefinition(
        case_id="native-tracing-fieldlines-ncsx",
        create_input=create_native_tracing_fieldlines_ncsx_input,
        execute=execute_native_tracing_fieldlines_ncsx,
    ),
    "native-tracing-fieldlines-qa": CaseDefinition(
        case_id="native-tracing-fieldlines-qa",
        create_input=create_native_tracing_fieldlines_qa_input,
        execute=execute_native_tracing_fieldlines_qa,
    ),
    "native-tracing-particle": CaseDefinition(
        case_id="native-tracing-particle",
        create_input=create_native_tracing_particle_input,
        execute=execute_native_tracing_particle,
    ),
    "native-stage-two-optimization-minimal": CaseDefinition(
        case_id="native-stage-two-optimization-minimal",
        create_input=create_native_stage_two_optimization_minimal_input,
        execute=execute_native_stage_two_optimization_minimal,
        # Upstream's run ends at its 300-iteration L-BFGS-B limit (status 1); the band admits that outcome at
        # native_default. The reduced scale converges inside the same cap, so no work budget is declared.
        quality_bands=(official_quality_band("native-stage-two-optimization-minimal"),),
    ),
    "native-stage-two-optimization": CaseDefinition(
        case_id="native-stage-two-optimization",
        create_input=create_native_stage_two_optimization_input,
        execute=execute_native_stage_two_optimization,
        work_budget_contract=WorkBudgetContract(
            scales=_FIXED_BUDGET_SCALES,
            derivation="Upstream standard stage-two optimization fixes L-BFGS-B MAXITER per stage; accepted endpoints may reach the cap.",
        ),
    ),
    "native-stage-two-optimization-finitebuild": CaseDefinition(
        case_id="native-stage-two-optimization-finitebuild",
        create_input=create_native_stage_two_finitebuild_input,
        execute=execute_native_stage_two_finitebuild,
        quality_bands=(
            official_quality_band("native-stage-two-optimization-finitebuild"),
        ),
        work_budget_contract=WorkBudgetContract(
            scales=("bounded",),
            derivation="Upstream finite-build optimization uses a fixed SciPy iteration cap; accepted endpoints may exhaust it. At native_default the endpoint quality band admits the same outcome.",
        ),
    ),
    "native-stage-two-optimization-planar-coils": CaseDefinition(
        case_id="native-stage-two-optimization-planar-coils",
        create_input=create_native_stage_two_optimization_planar_coils_input,
        execute=execute_native_stage_two_optimization_planar_coils,
        # At the bounded scale both lanes run each stage to MAXITER on a path that forks at
        # round-off (the lanes' end objectives differ by 6 %; their own one-ulp twins scatter as
        # far), so the end objective is informational there and the stages are judged from shared
        # states (PLAN.md amendment 5, P1): the objective and gradient at upstream's 28 recorded
        # bounded states against a derived rounding bound, and the iterates against the lanes' own
        # one-ulp twin envelope up to the horizon. At native_default the lanes' end objectives
        # agree to 1.5e-7 relative; the work budget and the lane-against-lane routes decide there.
        stage_wise_contracts=(
            StageWiseContract(
                case_id="native-stage-two-optimization-planar-coils",
                scale="bounded",
                informational_observables=("final:objective",),
                deciding_observables=(
                    "initial:objective",
                    "initial:objective_gradient",
                ),
                same_state_tests=(
                    "tests/integration/test_jax_mirror_planar_coils_bounded_upstream_states.py",
                    "tests/integration/test_jax_mirror_planar_coils_bounded_trajectory_twins.py",
                ),
                derivation=(
                    "PLAN.md amendment 5 part 1, P1 (registered 2026-09-29 16:24 EDT, before any "
                    "formal run): the upstream-scatter band at bounded is removed. Deciding: every "
                    "lane-pair route but final:objective (including first:objective, which P2 "
                    "passed), the x0 same-state routes, and two tracked tests on the jax-cpu lane: "
                    "the native and JAX objective and gradient at x0 and at the 27 states of "
                    "upstream's tracked bounded record within twice the derived first-order "
                    "rounding bound of the formula both evaluate (tests/planar_stage_two_roundoff.py "
                    "over tests/forward_roundoff_bound.py), and the native-versus-JAX iterate "
                    "distance within the envelope of the lanes' sixteen one-ulp twins (upstream's "
                    "draws k = 1..8) before the horizon at which every twin distance reaches 1e-6. "
                    "The jax-gpu lane is not covered by the two tests (their trig constants are "
                    "the host libraries'). Upstream 9e027eac3's own trajectory uses a stale "
                    "CurvePlanarFourier Jacobian (src/simsoptpp/curveplanarfourier.h:94-105), so "
                    "its end values cannot bound a correct-gradient optimizer"
                ),
            ),
        ),
        work_budget_contract=WorkBudgetContract(
            scales=_FIXED_BUDGET_SCALES,
            derivation="Upstream's own planar-coil run does NOT end at the cap: 9e027eac3 keeps the four CurvePlanarFourier Jacobians in the persistent cache (src/simsoptpp/curveplanarfourier.h:94-105), which is only sound for a curve linear in its dofs, so upstream's gradient is stale after the first evaluated state and L-BFGS-B stagnates at nit 135 and 69 (status 0, RELATIVE REDUCTION OF F <= FACTR*EPSMCH, nfev 604 and 844). A lane whose CurvePlanarFourier Jacobian follows the dofs instead runs the script's own MAXITER per stage, so a budget exit is this case's expected honest terminal status. The end point is therefore NOT compared with upstream's; what is compared with upstream is the objective VALUE at upstream's own states (bitwise at x0 and at both official end states) and the gradient at x0 (2.56e-16 relative), in tests/integration/test_jax_mirror_planar_coils_official_states.py.",
        ),
    ),
    "native-stage-two-optimization-stochastic": CaseDefinition(
        case_id="native-stage-two-optimization-stochastic",
        create_input=create_native_stage_two_optimization_stochastic_input,
        execute=execute_native_stage_two_optimization_stochastic,
        work_budget_contract=WorkBudgetContract(
            scales=_FIXED_BUDGET_SCALES,
            derivation="Upstream stochastic stage-two optimization uses a fixed outer iteration cap; accepted endpoints may exhaust it.",
        ),
    ),
    "native-strain-optimization": CaseDefinition(
        case_id="native-strain-optimization",
        create_input=create_native_strain_optimization_input,
        execute=execute_native_strain_optimization,
        work_budget_contract=WorkBudgetContract(
            scales=_FIXED_BUDGET_SCALES,
            derivation="Upstream strain optimization fixes L-BFGS-B MAXITER; an accepted endpoint may stop at that cap.",
        ),
    ),
    "native-wireframe-rcls-basic": CaseDefinition(
        case_id="native-wireframe-rcls-basic",
        create_input=create_native_wireframe_rcls_basic_input,
        execute=execute_native_wireframe_rcls_basic,
    ),
    "native-wireframe-gsco-modular": CaseDefinition(
        case_id="native-wireframe-gsco-modular",
        create_input=create_native_wireframe_gsco_modular_input,
        execute=execute_native_wireframe_gsco_modular,
        work_budget_contract=WorkBudgetContract(
            scales=GSCO_REDUCED_BUDGET_SCALES,
            derivation="Upstream GSCO accepts an explicit max_iter cap (wireframe_optimization.cpp:281 stop_last_iter); the official native_default and CI runs stop earlier on their own rule, so only the reduced bounded cap is ever reached.",
        ),
    ),
    "native-wireframe-gsco-multistep": CaseDefinition(
        case_id="native-wireframe-gsco-multistep",
        create_input=create_native_wireframe_gsco_multistep_input,
        execute=execute_native_wireframe_gsco_multistep,
    ),
    "native-wireframe-gsco-sector-saddle": CaseDefinition(
        case_id="native-wireframe-gsco-sector-saddle",
        create_input=create_native_wireframe_gsco_sector_saddle_input,
        execute=execute_native_wireframe_gsco_sector_saddle,
        work_budget_contract=WorkBudgetContract(
            scales=GSCO_REDUCED_BUDGET_SCALES,
            derivation="Upstream GSCO accepts an explicit max_iter cap (wireframe_optimization.cpp:281 stop_last_iter); the official native_default and CI runs stop earlier on their own rule, so only the reduced bounded cap is ever reached.",
        ),
    ),
    "native-wireframe-rcls-with-ports": CaseDefinition(
        case_id="native-wireframe-rcls-with-ports",
        create_input=create_native_wireframe_rcls_with_ports_input,
        execute=execute_native_wireframe_rcls_with_ports,
    ),
}


def get_case(case_id: str) -> CaseDefinition:
    """Return one statically registered parity case."""
    try:
        return _CASES[case_id]
    except KeyError as error:
        raise ValueError(f"unknown or unimplemented parity case: {case_id}") from error


def implemented_case_ids() -> tuple[str, ...]:
    """Return implemented case IDs in deterministic registry order."""
    return tuple(_CASES)
