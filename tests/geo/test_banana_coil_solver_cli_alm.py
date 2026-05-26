"""CLI plumbing tests for the Stage 2 banana coil solver ALM flags.

Covers the rollout of ``--alm-fix-signal-mismatch-guard`` (mirrors the
single-stage CLI of the same name in
``SINGLE_STAGE/single_stage_banana_example.py``). The flag must:

1. Default to OFF when neither the CLI flag nor the
   ``ALM_FIX_SIGNAL_MISMATCH_GUARD`` env var is set.
2. Flip ON when ``--alm-fix-signal-mismatch-guard`` is passed.
3. Propagate into ``ALMSettings.continue_on_signal_mismatch``
   via ``build_stage2_alm_settings`` (the canonical builder consumed by
   Stage 2 driver, ranker, and homotopy stepper paths).

See ``docs/alm_hybrid_signal_contract_2026-05-08.md`` (lines 37-41) for
the contract this flag aligns Stage 2 with.
"""

import sys
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = REPO_ROOT / "examples" / "single_stage_optimization"
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

from STAGE_2 import banana_coil_solver as stage2_solver  # noqa: E402
from banana_opt import stage2_objectives  # noqa: E402


def _parse(argv_tail, env_overrides):
    argv = ["banana_coil_solver.py", *argv_tail]
    # ``ALM_FIX_SIGNAL_MISMATCH_GUARD`` participates in the argparse default;
    # neutralize any inherited env so the CLI surface is tested in isolation.
    env = {"ALM_FIX_SIGNAL_MISMATCH_GUARD": "0", **env_overrides}
    with mock.patch.object(sys, "argv", argv), mock.patch.dict(
        "os.environ", env, clear=False
    ):
        return stage2_solver.parse_args()


class Stage2AlmFixSignalMismatchGuardTests(unittest.TestCase):
    """The opt-in rollout flag must default OFF and only flip ON when set."""

    def test_default_args_have_flag_off(self):
        args = _parse([], env_overrides={"ALM_FIX_SIGNAL_MISMATCH_GUARD": "0"})
        self.assertFalse(args.alm_fix_signal_mismatch_guard)

    def test_cli_flag_flips_flag_on(self):
        args = _parse(
            ["--alm-fix-signal-mismatch-guard"],
            env_overrides={"ALM_FIX_SIGNAL_MISMATCH_GUARD": "0"},
        )
        self.assertTrue(args.alm_fix_signal_mismatch_guard)

    def test_env_var_truthy_flips_flag_on(self):
        args = _parse([], env_overrides={"ALM_FIX_SIGNAL_MISMATCH_GUARD": "1"})
        self.assertTrue(args.alm_fix_signal_mismatch_guard)


class Stage2BuildAlmSettingsHonorsFlagTests(unittest.TestCase):
    """``build_stage2_alm_settings`` must propagate the CLI flag into ALMSettings."""

    def _parsed_args_with_flag(self, *, flag_set):
        argv_tail = ["--alm-fix-signal-mismatch-guard"] if flag_set else []
        return _parse(argv_tail, env_overrides={"ALM_FIX_SIGNAL_MISMATCH_GUARD": "0"})

    def test_default_propagates_false_into_alm_settings(self):
        args = self._parsed_args_with_flag(flag_set=False)
        settings = stage2_objectives.build_stage2_alm_settings(args)
        self.assertFalse(settings.continue_on_signal_mismatch)

    def test_opt_in_propagates_true_into_alm_settings(self):
        args = self._parsed_args_with_flag(flag_set=True)
        settings = stage2_objectives.build_stage2_alm_settings(args)
        self.assertTrue(settings.continue_on_signal_mismatch)


if __name__ == "__main__":
    unittest.main()
