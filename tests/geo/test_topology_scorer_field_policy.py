"""Contract tests for the ``--topology-scorer-field-policy`` flag.

The in-loop topology scorer defaults to ``field_policy="auto"``, which selects an
InterpolatedField for long traces (fast but optimistic). ``never`` forces the
exact Biot-Savart field in the in-loop scorer so the in-loop confinement signal
matches the exact judge. These tests pin (a) the argparse default is ``"auto"``,
(b) ``never`` is accepted and threaded, (c) the configured policy reaches the
``safe_score_topology`` call site through the module global, and (d) the default
``"auto"`` is byte-identical to the prior ``field_policy=None`` default at the
field-selection level (no behavior change when the flag is unset).
"""

import importlib.util
import inspect
import sys
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import numpy as np
import simsopt.field as simsopt_field

from geo._frontier_test_helpers import ensure_examples_import_path

ensure_examples_import_path()

import topology_scorer


EXAMPLE_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "single_stage_optimization"
    / "SINGLE_STAGE"
    / "single_stage_banana_example.py"
)


def load_single_stage_example_module():
    spec = importlib.util.spec_from_file_location(
        f"single_stage_banana_example_{uuid.uuid4().hex}",
        EXAMPLE_MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RingSurface:
    """Lightweight Boozer-surface stub (mirrors the WBA contract test).

    Only ``nfp`` and ``cross_section`` are exercised; ``prepare_topology_field``
    is monkeypatched away, so no InterpolatedField or real BiotSavart is built.
    """

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


class TopologyScorerFieldPolicyFlagTest(unittest.TestCase):
    def test_parse_args_default_field_policy_is_auto(self):
        # (a) Default (flag unset) must resolve to "auto" so default runs stay
        # byte-identical to the historical interpolated in-loop scorer.
        module = load_single_stage_example_module()

        with patch.object(
            sys,
            "argv",
            ["single_stage_banana_example.py"],
        ):
            args = module.parse_args()

        self.assertEqual(args.topology_scorer_field_policy, "auto")

    def test_parse_args_field_policy_never_is_accepted(self):
        # (b) "never" forces the exact Biot-Savart field in the in-loop scorer.
        module = load_single_stage_example_module()

        with patch.object(
            sys,
            "argv",
            [
                "single_stage_banana_example.py",
                "--topology-scorer-field-policy",
                "never",
            ],
        ):
            args = module.parse_args()

        self.assertEqual(args.topology_scorer_field_policy, "never")

    def test_parse_args_field_policy_rejects_unknown_choice(self):
        # The choices set is exactly ("auto", "never"); anything else fails closed.
        module = load_single_stage_example_module()

        with patch.object(
            sys,
            "argv",
            [
                "single_stage_banana_example.py",
                "--topology-scorer-field-policy",
                "always",
            ],
        ):
            with self.assertRaises(SystemExit):
                module.parse_args()

    def test_module_global_default_field_policy_is_auto(self):
        # The module-global mirror (used at import time before __main__ runs and
        # by the call site) defaults to "auto", matching the argparse default.
        module = load_single_stage_example_module()

        self.assertEqual(module.TOPOLOGY_SCORER_FIELD_POLICY, "auto")

    def test_field_policy_is_threaded_from_argparse_to_call_site(self):
        # (c) Non-tautological end-to-end threading, asserted on real source:
        #   1. __main__ reassigns the module global from the parsed arg, and
        #   2. the safe_score_topology call site reads that same global.
        # Together these tie the configured policy to the in-loop scorer call;
        # neither assertion holds if either link is dropped.
        module = load_single_stage_example_module()

        main_source = inspect.getsource(
            inspect.getmodule(module.parse_args)
        )
        self.assertIn(
            "TOPOLOGY_SCORER_FIELD_POLICY = args.topology_scorer_field_policy",
            main_source,
        )

        call_site_source = inspect.getsource(module.maybe_record_topology_score)
        self.assertIn("safe_score_topology(", call_site_source)
        self.assertIn(
            "field_policy=TOPOLOGY_SCORER_FIELD_POLICY",
            call_site_source,
        )

    def test_auto_is_byte_identical_to_legacy_none_default(self):
        # (d) Behavioral byte-identity: passing field_policy="auto" explicitly
        # must drive the exact same prepare_topology_field resolution as the
        # historical field_policy=None default. Capture what score_topology
        # forwards to prepare_topology_field for each, then compare.
        captured_policies = []

        def fake_prepare_topology_field(*_args, **kwargs):
            captured_policies.append(kwargs["field_policy"])
            return object(), {"selected_mode": "native"}

        with patch.object(
            topology_scorer,
            "build_stopping_criteria",
            lambda *_a, **_k: ([object()], ["surface_exit"]),
        ), patch.object(
            topology_scorer,
            "midplane_seed_radii",
            lambda *_a, **_k: np.array([1.0, 1.1]),
        ), patch.object(
            topology_scorer,
            "prepare_topology_field",
            fake_prepare_topology_field,
        ), patch.object(
            topology_scorer, "cross_section_span", lambda *_a: 1.0
        ), patch.object(
            simsopt_field,
            "compute_fieldlines",
            lambda *_a, **_k: (
                [np.array([[0.0, 1.0, 0.0, 0.0]])],
                [np.array([[0.4, 0.0, 1.0, 0.0, 0.0]])],
            ),
        ):
            common = dict(
                nfieldlines=2,
                tmax=2.0,
                tol=1e-7,
                nphis=1,
                compute_transport_diagnostics=False,
                compute_invariant_torus_classification=False,
            )
            topology_scorer.score_topology(
                RingSurface(), object(), field_policy=None, **common
            )
            topology_scorer.score_topology(
                RingSurface(), object(), field_policy="auto", **common
            )

        self.assertEqual(len(captured_policies), 2)
        legacy_none_resolved, explicit_auto_resolved = captured_policies
        self.assertEqual(legacy_none_resolved, "auto")
        self.assertEqual(explicit_auto_resolved, "auto")
        self.assertEqual(legacy_none_resolved, explicit_auto_resolved)


if __name__ == "__main__":
    unittest.main()
