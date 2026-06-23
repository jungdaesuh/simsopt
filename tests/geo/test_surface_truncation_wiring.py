"""Wiring tests for the HOLD-FIXED truncation option in the banana driver.

These exercise the driver's thin caller
``initialize_published_surface_data_from_stage2_seed`` -- specifically its
truncation DECISION logic, not any Boozer solve. The real (already tested)
``build_boozer_surface_family`` is replaced with a stub that returns a
controllable ``provenance.truncated``, and every other collaborator the caller
touches (seed validation, name resolution, outer-seed construction, the
contract/entry callables) is stubbed out, so no geometry or solver runs. The
strict published-name postcondition is replaced with a SPY so we can assert
exactly when it does and does not run.

Observable behaviors under test (all at the caller's public surface -- its
return value plus the arguments/calls it makes):

  * default (no truncation globals published): the caller asks
    ``build_boozer_surface_family`` for the strict policy
    (``allow_truncation=False``, ``min_surfaces=len(configs)``) and ALWAYS runs
    the strict inner0-based name postcondition -- the existing
    published_multisurface path is unchanged;
  * truncation ON + a truncated band (``provenance.truncated=True``): the caller
    passes ``allow_truncation=True`` with the published ``min_surfaces`` floor and
    SKIPS the strict name postcondition (the family helper already validated the
    outer-suffix band with its relaxed nested+ordered check);
  * truncation ON but the full stack was accepted (``provenance.truncated=False``):
    the caller still runs the strict postcondition.

The driver module is loaded once for the class. Because it is loaded as a
regular module (not ``__main__``), the ``if __name__ == "__main__":`` block that
publishes ``SURFACE_ALLOW_TRUNCATION`` / ``SURFACE_MIN_SURFACES`` does NOT run, so
absence-of-globals is the genuine default state under test; the truncation-ON
cases set those module globals explicitly and remove them again on cleanup.
"""

import importlib.util
import sys
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace

EXAMPLE_ROOT = (
    Path(__file__).resolve().parents[2] / "examples" / "single_stage_optimization"
)
SINGLE_STAGE_ENTRYPOINT_PATH = (
    EXAMPLE_ROOT / "SINGLE_STAGE" / "single_stage_banana_example.py"
)
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))


def _load_single_stage_example_module():
    """Import the banana driver as a normal module (so __main__ does not run)."""
    spec = importlib.util.spec_from_file_location(
        f"single_stage_banana_example_{uuid.uuid4().hex}",
        SINGLE_STAGE_ENTRYPOINT_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _StubProvenance:
    """Minimal stand-in for ``BoozerFamilyProvenance`` exposing only ``.truncated``.

    The caller reads exactly one attribute of the provenance object returned by
    ``build_boozer_surface_family`` -- ``truncated`` -- to decide whether to run
    the strict postcondition; nothing else is touched here.
    """

    def __init__(self, truncated):
        self.truncated = truncated


# Distinct sentinel the stubbed family helper returns as ``ordered_surface_data``;
# the caller must hand this straight back unchanged, so tests assert identity.
_SENTINEL_SURFACE_DATA = [{"name": "sentinel"}]


class TruncationWiringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_single_stage_example_module()

    def setUp(self):
        module = self.module
        # Three inner-to-outer published configs (inner0, inner1, outer); their
        # contents are irrelevant because the family helper is stubbed -- only the
        # COUNT (3) matters for the strict-mode ``min_surfaces=len(configs)`` check.
        self.surface_configs = [
            {"name": "inner0", "target_volume": 1.0},
            {"name": "inner1", "target_volume": 2.0},
            {"name": "outer", "target_volume": 3.0},
        ]
        self.ordered_names = ["inner0", "inner1", "outer"]

        # --- Records of the caller's observable decisions ------------------
        # The kwargs the caller passed to build_boozer_surface_family (we read
        # allow_truncation + min_surfaces back out of here).
        self.family_call_kwargs = {}
        # How many times the strict published-name postcondition was invoked, and
        # with what payload (proves it both ran/skipped AND saw the helper's data).
        self.strict_postcondition_calls = []

        # --- Stub every collaborator the caller touches --------------------
        # Seed validation + name resolution: no-op / canned, so the caller reaches
        # the family build with our 3 configs.
        self._patch(module, "_require_published_stage2_solved_seed", lambda seed: None)
        self._patch(
            module,
            "_published_surface_names",
            lambda count: list(self.ordered_names[:count]),
        )
        # Outer-seed construction + finite-G guard: trivial pass-throughs.
        self._patch(
            module,
            "BoozerFamilyOuterSeed",
            lambda *, surface, iota, G: SimpleNamespace(surface=surface, iota=iota, G=G),
        )
        self._patch(module, "_require_finite_explicit_G", lambda name, G: G)
        # Contract/entry callables only get forwarded into the (stubbed) family
        # helper, so identity stubs are enough.
        self._patch(
            module,
            "contract_surface_to_target_volume",
            lambda previous_surface, target_volume: previous_surface,
        )
        self._patch(
            module,
            "_surface_data_entry",
            lambda config, boozer_surface, provenance: {"name": config["name"]},
        )
        # Strict postcondition SPY: record each call instead of asserting names.
        self._patch(
            module,
            "_require_published_surface_data_postconditions",
            lambda data: self.strict_postcondition_calls.append(data),
        )

        # Stage-2 seed surface stand-in carrying the three attributes the caller
        # reads (.surface/.iota/.G); concrete values are inert under the stubs.
        self.stage2_seed_surface = SimpleNamespace(
            surface=object(), iota=0.3, G=1.0
        )

    def _patch(self, module, name, value):
        """Set ``module.name = value`` for one test and restore it on cleanup.

        Restores the original attribute (captured before patching) via
        ``addCleanup`` so monkeypatching cannot leak across tests even when an
        assertion fails mid-test.
        """
        original = getattr(module, name)
        setattr(module, name, value)
        self.addCleanup(setattr, module, name, original)

    def _install_family_stub(self, *, truncated):
        """Replace build_boozer_surface_family with a recording stub.

        The stub captures the kwargs it was called with (so the test can read back
        the truncation policy the caller chose) and returns the sentinel band plus
        a provenance whose ``truncated`` flag the test controls.
        """

        def fake_build_boozer_surface_family(ordered_configs, **kwargs):
            self.family_call_kwargs = dict(kwargs)
            return _SENTINEL_SURFACE_DATA, _StubProvenance(truncated)

        self._patch(
            self.module,
            "build_boozer_surface_family",
            fake_build_boozer_surface_family,
        )

    def _set_truncation_globals(self, *, allow, min_surfaces):
        """Publish the truncation policy globals the __main__ block would set.

        These module globals are ABSENT by default (the entrypoint guard does not
        run on import, which is exactly the default state the no-globals test
        relies on). They cannot be saved/restored like an existing attribute, so
        set them here and DELETE them on cleanup, returning the module to its
        clean absent state regardless of test outcome.
        """
        self._set_absent_global(self.module, "SURFACE_ALLOW_TRUNCATION", allow)
        self._set_absent_global(self.module, "SURFACE_MIN_SURFACES", min_surfaces)

    def _set_absent_global(self, module, name, value):
        """Set a module global that did not previously exist; delete it on cleanup."""
        assert not hasattr(module, name), (
            f"{name} must be absent by default (the __main__ guard should not run "
            "on import); a leaked global would invalidate the default-path test."
        )
        setattr(module, name, value)
        self.addCleanup(delattr, module, name)

    def _call(self):
        return self.module.initialize_published_surface_data_from_stage2_seed(
            self.surface_configs,
            mpol=4,
            ntor=4,
            bs=object(),
            constraint_weight=1.0,
            boozer_I=0.0,
            nfp=5,
            stage2_seed_surface=self.stage2_seed_surface,
        )

    def test_default_runs_strict_postcondition_and_requests_strict_policy(self):
        """No truncation globals -> strict policy + strict postcondition (unchanged path)."""
        self._install_family_stub(truncated=False)

        ordered_surface_data, warm_start_paths = self._call()

        # Caller asks the family helper for the strict policy: truncation off,
        # min_surfaces pinned to the requested stack size.
        self.assertFalse(self.family_call_kwargs["allow_truncation"])
        self.assertEqual(
            self.family_call_kwargs["min_surfaces"], len(self.surface_configs)
        )
        # Strict inner0-based name postcondition ran exactly once, on the band the
        # family helper returned.
        self.assertEqual(len(self.strict_postcondition_calls), 1)
        self.assertIs(self.strict_postcondition_calls[0], _SENTINEL_SURFACE_DATA)
        # Caller returns the helper's band unchanged plus the empty warm-start list.
        self.assertIs(ordered_surface_data, _SENTINEL_SURFACE_DATA)
        self.assertEqual(warm_start_paths, [])

    def test_truncated_band_skips_strict_postcondition(self):
        """Truncation ON + provenance.truncated -> strict postcondition SKIPPED."""
        self._set_truncation_globals(allow=True, min_surfaces=2)
        self._install_family_stub(truncated=True)

        ordered_surface_data, warm_start_paths = self._call()

        # Caller forwards the published truncation policy to the family helper.
        self.assertTrue(self.family_call_kwargs["allow_truncation"])
        self.assertEqual(self.family_call_kwargs["min_surfaces"], 2)
        # The strict inner0-based name postcondition must NOT run for a truncated
        # outer-suffix band (the helper already validated it with the relaxed check).
        self.assertEqual(self.strict_postcondition_calls, [])
        self.assertIs(ordered_surface_data, _SENTINEL_SURFACE_DATA)
        self.assertEqual(warm_start_paths, [])

    def test_truncation_on_but_full_acceptance_still_runs_strict_postcondition(self):
        """Truncation ON but provenance.truncated=False -> strict postcondition RUNS."""
        self._set_truncation_globals(allow=True, min_surfaces=2)
        self._install_family_stub(truncated=False)

        self._call()

        # allow_truncation was on, but because the full requested stack was accepted
        # (truncated=False) the band is still inner0-based, so the strict
        # postcondition runs exactly once.
        self.assertTrue(self.family_call_kwargs["allow_truncation"])
        self.assertEqual(len(self.strict_postcondition_calls), 1)
        self.assertIs(self.strict_postcondition_calls[0], _SENTINEL_SURFACE_DATA)


if __name__ == "__main__":
    unittest.main()
