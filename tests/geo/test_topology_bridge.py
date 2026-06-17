"""Unit tests for the Phase 3a non-Boozer field-line iota proxy module."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


EXAMPLE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "single_stage_optimization"
)
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))


def load_topology_bridge():
    return importlib.import_module("banana_opt.topology_bridge")


def _toroid_surface(rmax: float = 1.0, rmin: float = 0.7, zmax: float = 0.2):
    """Minimal duck-typed surface with the methods topology_bridge calls.

    The convergence sub-study and main proxy only use ``surface.gamma()``
    (for the seed radii + escape-cage envelope) and
    ``surface.cross_section(phi, thetas=...)`` (for the coordinate-axis
    centroid). Both are implemented in closed form for a circular ring
    so the test fixture is deterministic and independent of any heavy
    simsopt geometry construction.
    """
    nphi, ntheta = 16, 32

    def cross_section(phi: float, thetas: int):
        toroidal_angle = 2.0 * np.pi * float(phi)
        theta = np.linspace(0.0, 2.0 * np.pi, int(thetas), endpoint=False)
        center_r = 0.5 * (rmin + rmax)
        radius = 0.5 * (rmax - rmin)
        x = (center_r + radius * np.cos(theta)) * np.cos(toroidal_angle)
        y = (center_r + radius * np.cos(theta)) * np.sin(toroidal_angle)
        z = zmax * np.sin(theta)
        return np.stack([x, y, z], axis=-1)

    def gamma():
        phis = np.linspace(0.0, 1.0, nphi, endpoint=False)
        sections = [cross_section(phi, ntheta) for phi in phis]
        return np.stack(sections, axis=0)

    return SimpleNamespace(
        gamma=gamma,
        cross_section=cross_section,
    )


def _fake_history(nsteps: int = 4) -> np.ndarray:
    """Synthetic trajectory with the (t, x, y, z) shape simsopt returns."""
    t = np.linspace(0.0, 1.0, nsteps)
    return np.stack(
        [t, np.cos(t), np.sin(t), 0.1 * np.sin(t)],
        axis=-1,
    )


class TestPhase3aDataclassDistinctFromPhase3b:
    """N10: Phase 3a / Phase 3b dataclass names must NOT collide.

    The two bridges previously both exposed ``FieldlineIotaProxyResult``
    with incompatible field sets, so attribute access on a wrong-class
    instance crashed at runtime with ``AttributeError``. After the
    rename Phase 3a owns ``Phase3aFieldlineIotaProxyResult`` and Phase
    3b keeps ``FieldlineIotaProxyResult``; this test pins the contract
    so a future revert (or a backwards-compat alias) fails closed.
    """

    def test_phase3a_dataclass_uses_explicit_phase3a_prefix(self):
        bridge = load_topology_bridge()
        assert hasattr(bridge, "Phase3aFieldlineIotaProxyResult")
        assert "Phase3aFieldlineIotaProxyResult" in bridge.__all__

    def test_phase3a_does_not_expose_legacy_collision_name(self):
        """The bare ``FieldlineIotaProxyResult`` symbol is reserved for Phase 3b."""
        bridge = load_topology_bridge()
        assert not hasattr(bridge, "FieldlineIotaProxyResult")
        assert "FieldlineIotaProxyResult" not in bridge.__all__

    def test_phase3a_and_phase3b_dataclasses_are_distinct(self):
        bridge_3a = load_topology_bridge()
        from banana_opt import boozer_topology_bridge as bridge_3b
        # Phase 3a's result carries (iota_proxy, valid, n_surviving,
        # n_transits_used, escape_count, reason); Phase 3b's carries
        # (iota_proxy_mean, iota_proxy_std, iota_per_line,
        # n_lines_seeded, n_lines_survived, tmax, valid,
        # convergence_residual, invalid_reason). Without distinct names,
        # consumers accessing a Phase 3b field on a Phase 3a instance
        # (or vice-versa) crash at runtime.
        assert (
            bridge_3a.Phase3aFieldlineIotaProxyResult
            is not bridge_3b.FieldlineIotaProxyResult
        )
        phase3a_fields = {
            f.name
            for f in bridge_3a.Phase3aFieldlineIotaProxyResult.__dataclass_fields__.values()
        }
        phase3b_fields = {
            f.name
            for f in bridge_3b.FieldlineIotaProxyResult.__dataclass_fields__.values()
        }
        assert "iota_proxy" in phase3a_fields
        assert "iota_proxy_mean" in phase3b_fields
        assert "iota_proxy" not in phase3b_fields
        assert "iota_proxy_mean" not in phase3a_fields


class TestArtifactFields:
    def test_none_result_emits_four_contract_keys_all_none(self):
        bridge = load_topology_bridge()
        fields = bridge.fieldline_iota_proxy_artifact_fields(None)
        assert set(fields) == {
            "FIELDLINE_IOTA_PROXY",
            "FIELDLINE_IOTA_PROXY_VALID",
            "FIELDLINE_IOTA_PROXY_N_TRANSITS",
            "FIELDLINE_IOTA_PROXY_REASON",
        }
        assert all(value is None for value in fields.values())

    def test_valid_result_populates_all_keys(self):
        bridge = load_topology_bridge()
        result = bridge.Phase3aFieldlineIotaProxyResult(
            iota_proxy=0.2174,
            valid=True,
            n_surviving=5,
            n_transits_used=100,
            escape_count=0,
            reason=None,
        )
        fields = bridge.fieldline_iota_proxy_artifact_fields(result)
        assert fields["FIELDLINE_IOTA_PROXY"] == pytest.approx(0.2174)
        assert fields["FIELDLINE_IOTA_PROXY_VALID"] is True
        assert fields["FIELDLINE_IOTA_PROXY_N_TRANSITS"] == 100
        assert fields["FIELDLINE_IOTA_PROXY_REASON"] is None

    def test_invalid_result_persists_none_not_zero_for_iota(self):
        """Plan line 306: failure must surface as None, not fabricated 0.0."""
        bridge = load_topology_bridge()
        result = bridge.Phase3aFieldlineIotaProxyResult(
            iota_proxy=None,
            valid=False,
            n_surviving=2,
            n_transits_used=0,
            escape_count=3,
            reason=bridge.FAILURE_INSUFFICIENT_TRANSITS,
        )
        fields = bridge.fieldline_iota_proxy_artifact_fields(result)
        assert fields["FIELDLINE_IOTA_PROXY"] is None
        assert fields["FIELDLINE_IOTA_PROXY_VALID"] is False
        # Zero-transit cases must NOT report a fabricated transit count.
        assert fields["FIELDLINE_IOTA_PROXY_N_TRANSITS"] is None
        assert fields["FIELDLINE_IOTA_PROXY_REASON"] == bridge.FAILURE_INSUFFICIENT_TRANSITS


class TestParameterValidation:
    def test_nfieldlines_zero_rejected(self):
        bridge = load_topology_bridge()
        with pytest.raises(ValueError, match="nfieldlines must be positive"):
            bridge.compute_fieldline_iota_proxy(
                field=object(),
                surface=_toroid_surface(),
                nfieldlines=0,
                tmax=10.0,
                tol=1e-8,
                escape_radius=2.0,
                n_transits_target=10,
            )

    def test_tmax_zero_rejected(self):
        bridge = load_topology_bridge()
        with pytest.raises(ValueError, match="tmax must be positive"):
            bridge.compute_fieldline_iota_proxy(
                field=object(),
                surface=_toroid_surface(),
                nfieldlines=3,
                tmax=0.0,
                tol=1e-8,
                escape_radius=2.0,
                n_transits_target=10,
            )

    def test_tol_zero_rejected(self):
        bridge = load_topology_bridge()
        with pytest.raises(ValueError, match="tol must be positive"):
            bridge.compute_fieldline_iota_proxy(
                field=object(),
                surface=_toroid_surface(),
                nfieldlines=3,
                tmax=10.0,
                tol=0.0,
                escape_radius=2.0,
                n_transits_target=10,
            )

    def test_n_transits_target_zero_rejected(self):
        bridge = load_topology_bridge()
        with pytest.raises(ValueError, match="n_transits_target must be positive"):
            bridge.compute_fieldline_iota_proxy(
                field=object(),
                surface=_toroid_surface(),
                nfieldlines=3,
                tmax=10.0,
                tol=1e-8,
                escape_radius=2.0,
                n_transits_target=0,
            )

    def test_escape_radius_zero_rejected(self):
        bridge = load_topology_bridge()
        with pytest.raises(ValueError, match="escape_radius must be positive"):
            bridge.compute_fieldline_iota_proxy(
                field=object(),
                surface=_toroid_surface(),
                nfieldlines=3,
                tmax=10.0,
                tol=1e-8,
                escape_radius=0.0,
                n_transits_target=10,
            )


class TestProxyTracingContract:
    """Substitute the simsopt tracer to verify the surrounding contract.

    We deliberately do not stand up a real BiotSavart field for these
    tests — the simsopt tracer integration tests cover that. What we
    need here is to lock the failure-routing logic, the per-line
    classification (surviving vs escaped), and the dataclass payload
    construction. The tests inject deterministic histories + phi-hit
    arrays via monkeypatch so the proxy behavior is fully observable.
    """

    def test_all_lines_escape_returns_escaped_reason(self, monkeypatch):
        bridge = load_topology_bridge()
        n_lines = 3
        histories = [_fake_history() for _ in range(n_lines)]
        # idx < 0 in the final row marks "stopping criterion hit".
        phi_hits = [
            np.array([[0.0, -1.0, 0.5, 0.0, 0.0]]) for _ in range(n_lines)
        ]
        monkeypatch.setattr(
            bridge,
            "compute_fieldlines",
            lambda field, R0, Z0, **kwargs: (histories, phi_hits),
        )
        monkeypatch.setattr(
            bridge,
            "compute_toroidal_transits",
            lambda traced, flux=False: np.zeros(n_lines),
        )
        monkeypatch.setattr(
            bridge,
            "compute_poloidal_transits",
            lambda traced, ma=None, flux=True: np.zeros(n_lines),
        )

        result = bridge.compute_fieldline_iota_proxy(
            field=object(),
            surface=_toroid_surface(),
            nfieldlines=n_lines,
            tmax=10.0,
            tol=1e-8,
            escape_radius=2.0,
            n_transits_target=5,
        )
        assert result.valid is False
        assert result.iota_proxy is None
        assert result.reason == bridge.FAILURE_ESCAPED
        assert result.escape_count == n_lines

    def test_escaped_lines_above_transit_threshold_are_not_trusted(
        self, monkeypatch
    ):
        """Escaped trajectories must not contribute a fake converged iota."""
        bridge = load_topology_bridge()
        n_lines = 3
        histories = [_fake_history() for _ in range(n_lines)]
        phi_hits = [
            np.array([[0.0, -1.0, 0.5, 0.0, 0.0]]) for _ in range(n_lines)
        ]
        monkeypatch.setattr(
            bridge,
            "compute_fieldlines",
            lambda field, R0, Z0, **kwargs: (histories, phi_hits),
        )
        monkeypatch.setattr(
            bridge,
            "compute_toroidal_transits",
            lambda traced, flux=False: np.array([50.0, 100.0, 200.0]),
        )
        monkeypatch.setattr(
            bridge,
            "compute_poloidal_transits",
            lambda traced, ma=None, flux=True: np.array([0.0, 22.0, 44.0]),
        )

        result = bridge.compute_fieldline_iota_proxy(
            field=object(),
            surface=_toroid_surface(),
            nfieldlines=n_lines,
            tmax=10.0,
            tol=1e-8,
            escape_radius=2.0,
            n_transits_target=50,
        )
        assert result.valid is False
        assert result.iota_proxy is None
        assert result.reason == bridge.FAILURE_ESCAPED
        assert result.n_surviving == 0
        assert result.escape_count == n_lines

    def test_no_history_returns_no_history_reason(self, monkeypatch):
        bridge = load_topology_bridge()
        monkeypatch.setattr(
            bridge,
            "compute_fieldlines",
            lambda field, R0, Z0, **kwargs: ([], []),
        )
        # Transit functions should never be called when history is empty.
        monkeypatch.setattr(
            bridge,
            "compute_toroidal_transits",
            lambda traced, flux=False: (_ for _ in ()).throw(
                AssertionError("compute_toroidal_transits called on empty history")
            ),
        )
        monkeypatch.setattr(
            bridge,
            "compute_poloidal_transits",
            lambda traced, ma=None, flux=True: (_ for _ in ()).throw(
                AssertionError(
                    "compute_poloidal_transits called on empty history"
                )
            ),
        )
        result = bridge.compute_fieldline_iota_proxy(
            field=object(),
            surface=_toroid_surface(),
            nfieldlines=3,
            tmax=10.0,
            tol=1e-8,
            escape_radius=2.0,
            n_transits_target=5,
        )
        assert result.valid is False
        assert result.iota_proxy is None
        assert result.reason == bridge.FAILURE_NO_HISTORY

    def test_insufficient_transits_returns_failure(self, monkeypatch):
        """Surviving lines that haven't reached n_transits_target fail closed."""
        bridge = load_topology_bridge()
        n_lines = 4
        histories = [_fake_history() for _ in range(n_lines)]
        # All lines survived (no negative idx in final row).
        phi_hits = [
            np.array([[0.0, 0.0, 0.5, 0.0, 0.0]]) for _ in range(n_lines)
        ]
        monkeypatch.setattr(
            bridge,
            "compute_fieldlines",
            lambda field, R0, Z0, **kwargs: (histories, phi_hits),
        )
        # Each line completes only 3 toroidal transits but we asked for 10.
        monkeypatch.setattr(
            bridge,
            "compute_toroidal_transits",
            lambda traced, flux=False: np.full(n_lines, 3.0),
        )
        monkeypatch.setattr(
            bridge,
            "compute_poloidal_transits",
            lambda traced, ma=None, flux=True: np.full(n_lines, 1.0),
        )
        result = bridge.compute_fieldline_iota_proxy(
            field=object(),
            surface=_toroid_surface(),
            nfieldlines=n_lines,
            tmax=20.0,
            tol=1e-8,
            escape_radius=2.0,
            n_transits_target=10,
        )
        assert result.valid is False
        assert result.iota_proxy is None
        assert result.reason == bridge.FAILURE_INSUFFICIENT_TRANSITS
        assert result.n_surviving == n_lines
        assert result.escape_count == 0

    def test_clean_trace_returns_mean_iota_with_metadata(self, monkeypatch):
        bridge = load_topology_bridge()
        n_lines = 4
        histories = [_fake_history() for _ in range(n_lines)]
        # All lines survived to the tmax cap.
        phi_hits = [
            np.array([[0.0, 0.0, 0.5, 0.0, 0.0]]) for _ in range(n_lines)
        ]
        toroidal_counts = np.array([100.0, 100.0, 100.0, 100.0])
        poloidal_counts = np.array([22.0, 21.0, 22.0, 19.0])
        monkeypatch.setattr(
            bridge,
            "compute_fieldlines",
            lambda field, R0, Z0, **kwargs: (histories, phi_hits),
        )
        monkeypatch.setattr(
            bridge,
            "compute_toroidal_transits",
            lambda traced, flux=False: toroidal_counts,
        )
        monkeypatch.setattr(
            bridge,
            "compute_poloidal_transits",
            lambda traced, ma=None, flux=True: poloidal_counts,
        )
        result = bridge.compute_fieldline_iota_proxy(
            field=object(),
            surface=_toroid_surface(),
            nfieldlines=n_lines,
            tmax=300.0,
            tol=1e-8,
            escape_radius=2.0,
            n_transits_target=100,
        )
        expected = float(np.mean(poloidal_counts / toroidal_counts))
        assert result.valid is True
        assert result.iota_proxy == pytest.approx(expected)
        assert result.n_surviving == n_lines
        assert result.n_transits_used == 100
        assert result.escape_count == 0
        assert result.reason is None

    def test_partial_survival_uses_only_qualifying_lines(self, monkeypatch):
        """Lines below the transit threshold drop from the iota mean."""
        bridge = load_topology_bridge()
        n_lines = 4
        histories = [_fake_history() for _ in range(n_lines)]
        # Two lines survived, two escaped.
        phi_hits = [
            np.array([[0.0, 0.0, 0.5, 0.0, 0.0]]),
            np.array([[0.0, -1.0, 0.5, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 0.5, 0.0, 0.0]]),
            np.array([[0.0, -1.0, 0.5, 0.0, 0.0]]),
        ]
        toroidal_counts = np.array([100.0, 2.0, 120.0, 3.0])
        poloidal_counts = np.array([22.0, 0.0, 25.0, 0.0])
        monkeypatch.setattr(
            bridge,
            "compute_fieldlines",
            lambda field, R0, Z0, **kwargs: (histories, phi_hits),
        )
        monkeypatch.setattr(
            bridge,
            "compute_toroidal_transits",
            lambda traced, flux=False: toroidal_counts,
        )
        monkeypatch.setattr(
            bridge,
            "compute_poloidal_transits",
            lambda traced, ma=None, flux=True: poloidal_counts,
        )
        result = bridge.compute_fieldline_iota_proxy(
            field=object(),
            surface=_toroid_surface(),
            nfieldlines=n_lines,
            tmax=200.0,
            tol=1e-8,
            escape_radius=2.0,
            n_transits_target=100,
        )
        # Only lines 0 and 2 satisfy n_transits >= 100.
        expected = float(np.mean([22.0 / 100.0, 25.0 / 120.0]))
        assert result.valid is True
        assert result.iota_proxy == pytest.approx(expected)
        assert result.n_surviving == 2  # lines 0 and 2
        assert result.escape_count == 2  # lines 1 and 3

    def test_non_finite_transits_dropped(self, monkeypatch):
        bridge = load_topology_bridge()
        n_lines = 3
        histories = [_fake_history() for _ in range(n_lines)]
        phi_hits = [
            np.array([[0.0, 0.0, 0.5, 0.0, 0.0]]) for _ in range(n_lines)
        ]
        toroidal_counts = np.array([100.0, np.nan, 110.0])
        poloidal_counts = np.array([20.0, np.nan, 22.0])
        monkeypatch.setattr(
            bridge,
            "compute_fieldlines",
            lambda field, R0, Z0, **kwargs: (histories, phi_hits),
        )
        monkeypatch.setattr(
            bridge,
            "compute_toroidal_transits",
            lambda traced, flux=False: toroidal_counts,
        )
        monkeypatch.setattr(
            bridge,
            "compute_poloidal_transits",
            lambda traced, ma=None, flux=True: poloidal_counts,
        )
        result = bridge.compute_fieldline_iota_proxy(
            field=object(),
            surface=_toroid_surface(),
            nfieldlines=n_lines,
            tmax=300.0,
            tol=1e-8,
            escape_radius=2.0,
            n_transits_target=100,
        )
        # NaN entries must be dropped from the iota mean. ``n_surviving``
        # tracks escape-cage status, not transit finiteness, so all 3
        # lines remain "surviving" (none tripped a stopping criterion).
        assert result.valid is True
        assert result.n_surviving == 3
        assert result.iota_proxy == pytest.approx(
            float(np.mean([20.0 / 100.0, 22.0 / 110.0]))
        )


class TestEscapeCageConstruction:
    """The escape cage must be a non-empty list of SIMSOPT stop criteria."""

    def test_default_escape_radius_builds_four_criteria(self):
        bridge = load_topology_bridge()
        cage = bridge._build_escape_cage(_toroid_surface(), escape_radius=2.0)
        assert len(cage) == 4
        # All entries are SIMSOPT stopping criteria (have a C++ ``__call__``
        # under the hood); duck-type-check via the simsopt class names.
        from simsopt.field import (
            MaxRStoppingCriterion,
            MaxZStoppingCriterion,
            MinRStoppingCriterion,
            MinZStoppingCriterion,
        )
        criterion_types = {type(criterion).__name__ for criterion in cage}
        assert MaxRStoppingCriterion.__name__ in criterion_types
        assert MinRStoppingCriterion.__name__ in criterion_types
        assert MaxZStoppingCriterion.__name__ in criterion_types
        assert MinZStoppingCriterion.__name__ in criterion_types

    def test_zero_escape_radius_raises(self):
        bridge = load_topology_bridge()
        with pytest.raises(ValueError, match="escape_radius must be positive"):
            bridge._build_escape_cage(_toroid_surface(), escape_radius=0.0)


class TestSyntheticNonBootableField:
    """An escape-radius cage must catch a synthetic wandering field.

    We monkeypatch the SIMSOPT tracer to return the pathological
    behavior expected of a non-bootable field (all lines hit a stopping
    criterion immediately with zero toroidal transits) and assert that
    the proxy correctly fails closed rather than reporting a fabricated
    zero iota.
    """

    def test_non_bootable_field_fails_closed_with_iota_none(self, monkeypatch):
        bridge = load_topology_bridge()
        n_lines = 5
        # All lines tripped a stopping criterion at t=0.001 with zero transits.
        histories = [
            np.array([[0.0, 1.0, 0.0, 0.0], [0.001, 1.5, 0.0, 0.0]])
            for _ in range(n_lines)
        ]
        phi_hits = [
            np.array([[0.001, -1.0, 1.5, 0.0, 0.0]]) for _ in range(n_lines)
        ]
        monkeypatch.setattr(
            bridge,
            "compute_fieldlines",
            lambda field, R0, Z0, **kwargs: (histories, phi_hits),
        )
        monkeypatch.setattr(
            bridge,
            "compute_toroidal_transits",
            lambda traced, flux=False: np.zeros(n_lines),
        )
        monkeypatch.setattr(
            bridge,
            "compute_poloidal_transits",
            lambda traced, ma=None, flux=True: np.zeros(n_lines),
        )
        result = bridge.compute_fieldline_iota_proxy(
            field=object(),
            surface=_toroid_surface(),
            nfieldlines=n_lines,
            tmax=50.0,
            tol=1e-8,
            escape_radius=1.5,
            n_transits_target=10,
        )
        assert result.valid is False
        assert result.iota_proxy is None  # NOT 0.0
        assert result.escape_count == n_lines
        # Artifact field equally must persist None.
        fields = bridge.fieldline_iota_proxy_artifact_fields(result)
        assert fields["FIELDLINE_IOTA_PROXY"] is None
        assert fields["FIELDLINE_IOTA_PROXY_VALID"] is False


class TestFailureGuardReachability:
    """Each ``FAILURE_*`` tag in ``__all__`` must be produced by some input.

    Crucible finding F2 traced an unreachable ``FAILURE_NO_SURVIVING_LINES``
    branch caused by an earlier guard returning first. After the fix the
    failure-tag set is :data:`FAILURE_NO_HISTORY`, :data:`FAILURE_ESCAPED`,
    :data:`FAILURE_INSUFFICIENT_TRANSITS`, and
    :data:`FAILURE_TRACER_EXCEPTION` — every tag in ``__all__`` is reachable
    by at least one of the tests below or in :class:`TestSafeWrapper`.
    """

    def test_failure_tags_in_all_are_reachable(self):
        bridge = load_topology_bridge()
        # Sanity check the public failure-tag surface; FAILURE_NO_SURVIVING_LINES
        # was removed by the F2 fix because it is structurally subsumed by
        # FAILURE_ESCAPED (n_surviving == 0 implies escape_count == nfieldlines).
        expected_names = {
            "FAILURE_ESCAPED",
            "FAILURE_INSUFFICIENT_TRANSITS",
            "FAILURE_NO_HISTORY",
            "FAILURE_TRACER_EXCEPTION",
        }
        assert expected_names <= set(bridge.__all__)
        assert "FAILURE_NO_SURVIVING_LINES" not in bridge.__all__
        # Every exported failure name must resolve to a non-empty string tag.
        for name in expected_names:
            assert isinstance(getattr(bridge, name), str)
            assert getattr(bridge, name) != ""


class TestLineWasLost:
    """A line with zero phi crossings must be treated as lost.

    Crucible finding (Agent #2): the original ``_line_was_lost`` returned
    False for an empty hit array, silently inflating ``n_surviving`` even
    though no toroidal transit could be counted on that line. The fix
    counts the zero-hits case as lost so survival bookkeeping is monotone
    with the iota mean.
    """

    def test_empty_hits_counts_as_lost(self):
        bridge = load_topology_bridge()
        assert bridge._line_was_lost(np.zeros((0,))) is True
        assert bridge._line_was_lost(np.zeros((0, 5))) is True

    def test_stopping_criterion_hit_counts_as_lost(self):
        bridge = load_topology_bridge()
        assert bridge._line_was_lost(
            np.array([[0.0, -1.0, 0.5, 0.0, 0.0]])
        ) is True

    def test_clean_surface_crossing_counts_as_surviving(self):
        bridge = load_topology_bridge()
        assert bridge._line_was_lost(
            np.array([[0.0, 0.0, 0.5, 0.0, 0.0]])
        ) is False


class TestSafeWrapper:
    """The safe wrapper is the only swallow point per the no-defensive-try guideline.

    Crucible finding F3: Phase 3a was previously called raw at the
    artifact-pipeline integration site, so a simsoptpp ``RuntimeError`` on a
    numerically pathological field would crash the parent solver after the
    optimization had already completed. The safe wrapper now catches
    ``(ValueError, RuntimeError)`` narrowly and returns a
    :class:`Phase3aFieldlineIotaProxyResult` with ``valid=False`` and
    ``reason=FAILURE_TRACER_EXCEPTION`` so the artifact records the failure
    tag rather than failing closed via a process crash.
    """

    def test_safe_wrapper_passes_through_on_clean_input(self, monkeypatch):
        bridge = load_topology_bridge()
        n_lines = 3
        histories = [_fake_history() for _ in range(n_lines)]
        phi_hits = [
            np.array([[0.0, 0.0, 0.5, 0.0, 0.0]]) for _ in range(n_lines)
        ]
        toroidal_counts = np.full(n_lines, 100.0)
        poloidal_counts = np.full(n_lines, 22.0)
        monkeypatch.setattr(
            bridge,
            "compute_fieldlines",
            lambda field, R0, Z0, **kwargs: (histories, phi_hits),
        )
        monkeypatch.setattr(
            bridge,
            "compute_toroidal_transits",
            lambda traced, flux=False: toroidal_counts,
        )
        monkeypatch.setattr(
            bridge,
            "compute_poloidal_transits",
            lambda traced, ma=None, flux=True: poloidal_counts,
        )
        result = bridge.safe_compute_fieldline_iota_proxy(
            field=object(),
            surface=_toroid_surface(),
            nfieldlines=n_lines,
            tmax=200.0,
            tol=1e-8,
            escape_radius=2.0,
            n_transits_target=100,
        )
        assert result.valid is True
        assert result.iota_proxy == pytest.approx(0.22)
        assert result.reason is None

    def test_safe_wrapper_catches_runtime_error(self, monkeypatch):
        bridge = load_topology_bridge()

        def raise_runtime(*_args, **_kwargs):
            raise RuntimeError("simsoptpp numerical breakdown")

        monkeypatch.setattr(bridge, "compute_fieldlines", raise_runtime)
        result = bridge.safe_compute_fieldline_iota_proxy(
            field=object(),
            surface=_toroid_surface(),
            nfieldlines=3,
            tmax=10.0,
            tol=1e-8,
            escape_radius=2.0,
            n_transits_target=10,
        )
        assert result.valid is False
        assert result.iota_proxy is None
        assert result.reason == bridge.FAILURE_TRACER_EXCEPTION
        # The artifact-key payload must serialize the failure tag so
        # downstream donor-ranking tools can triage it.
        fields = bridge.fieldline_iota_proxy_artifact_fields(result)
        assert fields["FIELDLINE_IOTA_PROXY"] is None
        assert fields["FIELDLINE_IOTA_PROXY_VALID"] is False
        assert fields["FIELDLINE_IOTA_PROXY_REASON"] == bridge.FAILURE_TRACER_EXCEPTION

    def test_safe_wrapper_catches_value_error_from_validation(self):
        bridge = load_topology_bridge()
        # Negative escape_radius raises ValueError inside compute_fieldline_iota_proxy.
        result = bridge.safe_compute_fieldline_iota_proxy(
            field=object(),
            surface=_toroid_surface(),
            nfieldlines=3,
            tmax=10.0,
            tol=1e-8,
            escape_radius=-1.0,
            n_transits_target=10,
        )
        assert result.valid is False
        assert result.iota_proxy is None
        assert result.reason == bridge.FAILURE_TRACER_EXCEPTION

    def test_safe_wrapper_catches_cross_section_monotonicity_failure(self):
        bridge = load_topology_bridge()

        def cross_section(*_args, **_kwargs):
            raise Exception(
                "An error occured during calculation of the cross section. "
                "This happens when a surface 'goes back' on itself."
            )

        surface = SimpleNamespace(
            gamma=_toroid_surface().gamma,
            cross_section=cross_section,
        )
        result = bridge.safe_compute_fieldline_iota_proxy(
            field=object(),
            surface=surface,
            nfieldlines=3,
            tmax=10.0,
            tol=1e-8,
            escape_radius=2.0,
            n_transits_target=10,
        )
        assert result.valid is False
        assert result.iota_proxy is None
        assert result.reason == bridge.FAILURE_TRACER_EXCEPTION

    def test_safe_wrapper_does_not_swallow_unrelated_exceptions(self, monkeypatch):
        bridge = load_topology_bridge()

        def raise_type_error(*_args, **_kwargs):
            raise TypeError("genuine bug")

        monkeypatch.setattr(bridge, "compute_fieldlines", raise_type_error)
        with pytest.raises(TypeError, match="genuine bug"):
            bridge.safe_compute_fieldline_iota_proxy(
                field=object(),
                surface=_toroid_surface(),
                nfieldlines=3,
                tmax=10.0,
                tol=1e-8,
                escape_radius=2.0,
                n_transits_target=10,
            )


class TestSharedCoordinateAxis:
    """The centroid axis class lives in topology_bridge_shared.

    F13 fix: both Phase 3a and Phase 3b import the class from the shared
    module so the two bridges describe the same magnetic-axis proxy.
    """

    def test_shared_class_is_imported_into_phase3a(self):
        bridge = load_topology_bridge()
        from banana_opt import topology_bridge_shared as shared
        # The legacy ``_SurfaceCentroidCoordinateAxis`` alias resolves to
        # the SSOT class in the shared module.
        assert bridge._SurfaceCentroidCoordinateAxis is shared.SurfaceCentroidCoordinateAxis

    def test_shared_factory_returns_shared_class(self):
        from banana_opt import topology_bridge_shared as shared
        axis = shared.build_surface_centroid_axis(
            _toroid_surface(), nphi=16, ntheta=16
        )
        assert isinstance(axis, shared.SurfaceCentroidCoordinateAxis)
        # ``gamma_impl`` writes into a (1, 3) buffer in place.
        buf = np.zeros((1, 3))
        axis.gamma_impl(buf, 0.25)
        assert np.all(np.isfinite(buf))

    def test_shared_defaults_match_phase3a_constants(self):
        bridge = load_topology_bridge()
        from banana_opt import topology_bridge_shared as shared
        # SSOT: the constants in topology_bridge re-export the shared ones.
        assert bridge.DEFAULT_TMAX == shared.DEFAULT_TMAX
        assert bridge.DEFAULT_TOL == shared.DEFAULT_TOL
        assert bridge.DEFAULT_NFIELDLINES == shared.DEFAULT_NFIELDLINES


class TestConvergenceSweepFixedPopulation:
    """The convergence sub-study must measure trajectory drift, not population drift.

    Crucible finding F9: when the convergence script swept
    ``n_transits_target`` per rung, the per-line trusted population
    *changed* between rungs (lines admitted at N=100 were dropped at
    N=200 if their toroidal-transit count fell between the two
    thresholds). The acceptance criterion
    ``|iota_FL(N=200) - iota_FL(N=100)| < 0.005`` (plan line 274)
    assumes a fixed population whose trajectories are merely traced
    longer. The fix holds ``n_transits_target`` fixed at the lowest rung
    and scales only ``tmax`` per rung. The contract this test pins is
    that :func:`compute_fieldline_iota_proxy`, called with the same
    ``n_transits_target`` and the same per-line toroidal/poloidal
    counts but different ``tmax``, returns the same mean over the same
    trusted population.
    """

    def test_iota_mean_stable_across_tmax_when_population_fixed(self, monkeypatch):
        bridge = load_topology_bridge()
        n_lines = 5
        histories = [_fake_history() for _ in range(n_lines)]
        phi_hits = [
            np.array([[0.0, 0.0, 0.5, 0.0, 0.0]]) for _ in range(n_lines)
        ]
        # Every seed clears 60 toroidal transits (above the fixed floor)
        # and the per-line iotas are constant — a converged trajectory.
        toroidal_counts = np.full(n_lines, 60.0)
        poloidal_counts = np.array([12.0, 13.0, 12.0, 13.0, 12.0])

        monkeypatch.setattr(
            bridge,
            "compute_fieldlines",
            lambda field, R0, Z0, **kwargs: (histories, phi_hits),
        )
        monkeypatch.setattr(
            bridge,
            "compute_toroidal_transits",
            lambda traced, flux=False: toroidal_counts,
        )
        monkeypatch.setattr(
            bridge,
            "compute_poloidal_transits",
            lambda traced, ma=None, flux=True: poloidal_counts,
        )

        # Two rungs, same fixed transit floor (50), different tmax.
        result_short = bridge.compute_fieldline_iota_proxy(
            field=object(),
            surface=_toroid_surface(),
            nfieldlines=n_lines,
            tmax=500.0,
            tol=1e-8,
            escape_radius=2.0,
            n_transits_target=50,
        )
        result_long = bridge.compute_fieldline_iota_proxy(
            field=object(),
            surface=_toroid_surface(),
            nfieldlines=n_lines,
            tmax=1000.0,
            tol=1e-8,
            escape_radius=2.0,
            n_transits_target=50,
        )
        assert result_short.valid is True
        assert result_long.valid is True
        # Same population of trusted lines → identical iota means.
        assert result_short.n_surviving == n_lines
        assert result_long.n_surviving == n_lines
        assert result_short.iota_proxy == pytest.approx(result_long.iota_proxy)
        # Residual on a fixed-population converged trajectory is zero.
        assert abs(result_long.iota_proxy - result_short.iota_proxy) < 1.0e-12

    def test_varying_threshold_can_admit_different_populations(self, monkeypatch):
        """Counterexample: when the threshold varies, the mean shifts.

        This is the bug the F9 fix avoids — sweeping
        ``n_transits_target`` per rung changes which lines contribute to
        the mean, so the residual is conflated with the population
        change. Locked here as a contract so a future regression that
        re-introduces a per-rung threshold can be caught by the failure
        of the convergence sub-study to give a stable mean.
        """
        bridge = load_topology_bridge()
        n_lines = 4
        histories = [_fake_history() for _ in range(n_lines)]
        phi_hits = [
            np.array([[0.0, 0.0, 0.5, 0.0, 0.0]]) for _ in range(n_lines)
        ]
        # Two lines clear 60 transits (in both rungs), two only clear 110
        # (dropped from the N=200 rung). Per-line iotas differ between
        # the two subpopulations so the means differ.
        toroidal_counts = np.array([60.0, 60.0, 110.0, 110.0])
        poloidal_counts = np.array([6.0, 7.0, 33.0, 32.0])

        monkeypatch.setattr(
            bridge,
            "compute_fieldlines",
            lambda field, R0, Z0, **kwargs: (histories, phi_hits),
        )
        monkeypatch.setattr(
            bridge,
            "compute_toroidal_transits",
            lambda traced, flux=False: toroidal_counts,
        )
        monkeypatch.setattr(
            bridge,
            "compute_poloidal_transits",
            lambda traced, ma=None, flux=True: poloidal_counts,
        )

        result_low_threshold = bridge.compute_fieldline_iota_proxy(
            field=object(),
            surface=_toroid_surface(),
            nfieldlines=n_lines,
            tmax=2000.0,
            tol=1e-8,
            escape_radius=2.0,
            n_transits_target=50,
        )
        result_high_threshold = bridge.compute_fieldline_iota_proxy(
            field=object(),
            surface=_toroid_surface(),
            nfieldlines=n_lines,
            tmax=2000.0,
            tol=1e-8,
            escape_radius=2.0,
            n_transits_target=100,
        )
        assert result_low_threshold.valid is True
        assert result_high_threshold.valid is True
        # Means differ because the high-threshold rung admits only the
        # large-iota subpopulation; the fixed-population rung blends both.
        assert abs(
            result_high_threshold.iota_proxy - result_low_threshold.iota_proxy
        ) > 0.05
