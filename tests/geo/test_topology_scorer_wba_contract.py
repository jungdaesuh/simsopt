import numpy as np

from geo._frontier_test_helpers import ensure_examples_import_path

ensure_examples_import_path()

import topology_scorer


class RingSurface:
    nfp = 1

    def cross_section(self, *, phi, thetas):
        theta = np.linspace(0.0, 2.0 * np.pi, int(thetas), endpoint=False)
        return np.column_stack(
            (
                1.1 + 0.1 * np.cos(theta),
                np.zeros_like(theta) + float(phi),
                0.1 * np.sin(theta),
            )
        )


def test_invariant_torus_classification_reports_not_evaluated_for_short_traces():
    short_hits = np.array(
        [[float(step), 0.0, 1.1, 0.0, 0.0] for step in range(60)],
        dtype=float,
    )

    result = topology_scorer.invariant_torus_classification(
        [short_hits],
        RingSurface(),
    )

    assert result["invariant_torus_fraction"] is None
    assert result["wba_classified_seed_count"] == 0
    assert result["wba_classification_counts"]["insufficient_returns"] == 1
    assert result["wba_evaluation_state"] == "not_evaluated_insufficient_returns"


def test_empty_topology_score_does_not_promote_not_evaluated_wba_to_kam_fraction():
    result = topology_scorer.empty_topology_score_result(12, 50.0)

    assert result["invariant_torus_fraction"] is None
    assert result["kam_fraction"] is None
    assert result["kam_fraction_semantics"] is None
    assert result["wba_evaluation_state"] == "not_evaluated_no_classified_seeds"
