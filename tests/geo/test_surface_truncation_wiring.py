"""Wiring tests for the HOLD-FIXED truncation option in the banana driver.

These exercise the driver's thin caller
``initialize_published_surface_data_from_stage2_seed`` and its dispatcher
``initialize_surface_data_for_contract`` -- specifically the truncation DECISION
and POLICY-PROPAGATION logic, not any Boozer solve. The real (already tested)
``build_boozer_surface_family`` is replaced with a stub that returns a
controllable ``provenance.truncated`` and records the policy kwargs it was called
with; every other collaborator the caller touches (seed validation, name
resolution, outer-seed construction, the contract/entry callables, both
postcondition checks) is stubbed/spied, so no geometry or solver runs.

The truncation policy is an EXPLICIT keyword argument (``allow_truncation`` /
``min_surfaces``) threaded from the single setup-time call site
``initialize_surface_data_for_contract(...)`` in ``__main__`` down into the
published-family builder -- there is no module global, so the policy cannot be
read before it is set. (That was the silent-no-op bug: the policy was published
via ``globals()`` ~400 lines AFTER the setup-time build had already read it, so
``--surface-allow-truncation`` never reached the build. Threading it as an arg
makes that ordering bug structurally impossible and directly testable.)

Observable behaviors under test (all at the callers' public surface -- their
return value plus the arguments/calls they make):

  * default (``allow_truncation=False``): the caller asks
    ``build_boozer_surface_family`` for the strict policy with ``min_surfaces``
    pinned to the full requested stack (an explicit floor is ignored while off)
    and ALWAYS runs the strict inner0-based name postcondition -- the existing
    published_multisurface path is unchanged;
  * truncation ON + a truncated band (``provenance.truncated=True``): the caller
    forwards ``allow_truncation=True`` with the requested ``min_surfaces`` floor,
    SKIPS the strict name postcondition (the family helper already validated the
    outer-suffix band with its relaxed nested+ordered check) but STILL enforces
    the name-agnostic G-consistency check on that band;
  * truncation ON but the full stack was accepted (``provenance.truncated=False``):
    the caller still runs the full strict postcondition;
  * ``initialize_surface_data_for_contract`` forwards the policy into the
    published path -- the propagation hop whose absence made the flag a no-op.
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
    ``build_boozer_surface_family`` -- ``truncated`` -- to decide which
    postcondition to run; nothing else is touched here.
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
        # COUNT (3) matters for the strict-mode ``min_surfaces=len(configs)`` pin.
        self.surface_configs = [
            {"name": "inner0", "target_volume": 1.0},
            {"name": "inner1", "target_volume": 2.0},
            {"name": "outer", "target_volume": 3.0},
        ]
        self.ordered_names = ["inner0", "inner1", "outer"]

        # --- Records of the callers' observable decisions ------------------
        # The kwargs the caller passed to build_boozer_surface_family (we read
        # allow_truncation + min_surfaces back out of here).
        self.family_call_kwargs = {}
        # Each postcondition is a SPY recording its payload, so a test can assert
        # exactly which check ran (and that it saw the helper's band) without any
        # real name/volume/G validation running on the sentinel.
        self.strict_postcondition_calls = []
        self.g_consistency_calls = []

        # --- Stub every collaborator the caller touches --------------------
        self._patch(module, "_require_published_stage2_solved_seed", lambda seed: None)
        self._patch(
            module,
            "_published_surface_names",
            lambda count: list(self.ordered_names[:count]),
        )
        self._patch(
            module,
            "BoozerFamilyOuterSeed",
            lambda *, surface, iota, G: SimpleNamespace(surface=surface, iota=iota, G=G),
        )
        self._patch(module, "_require_finite_explicit_G", lambda name, G: G)
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
        # Both postconditions are spies (record instead of validate). The strict one
        # internally calls G-consistency in production, but it is stubbed out here, so
        # the g_consistency spy fires ONLY when the caller invokes it directly (the
        # truncated-band path) -- which is exactly what the truncated test asserts.
        self._patch(
            module,
            "_require_published_surface_data_postconditions",
            lambda data, **kwargs: self.strict_postcondition_calls.append(
                (data, dict(kwargs))
            ),
        )
        self._patch(
            module,
            "_require_published_G_consistency",
            lambda data: self.g_consistency_calls.append(data),
        )

        # Stage-2 seed surface stand-in carrying the three attributes the caller
        # reads (.surface/.iota/.G); concrete values are inert under the stubs.
        self.stage2_seed_surface = SimpleNamespace(surface=object(), iota=0.3, G=1.0)

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

        Captures the kwargs it was called with (so the test can read back the
        truncation policy the caller forwarded) and returns the sentinel band plus
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

    def _call(
        self,
        *,
        allow_truncation=False,
        min_surfaces=3,
        relax_initial_nesting=False,
    ):
        return self.module.initialize_published_surface_data_from_stage2_seed(
            self.surface_configs,
            mpol=4,
            ntor=4,
            bs=object(),
            constraint_weight=1.0,
            boozer_I=0.0,
            nfp=5,
            stage2_seed_surface=self.stage2_seed_surface,
            allow_truncation=allow_truncation,
            min_surfaces=min_surfaces,
            relax_initial_nesting=relax_initial_nesting,
        )

    def test_default_off_pins_min_surfaces_and_runs_strict_postcondition(self):
        """allow_truncation=False -> strict policy (min_surfaces pinned to the full
        stack, ignoring the passed floor) + strict postcondition; unchanged path."""
        self._install_family_stub(truncated=False)

        # Pass a bogus min_surfaces=99 to prove the off-branch PINS it to len(configs).
        ordered_surface_data, warm_start_paths = self._call(
            allow_truncation=False, min_surfaces=99
        )

        self.assertFalse(self.family_call_kwargs["allow_truncation"])
        self.assertEqual(
            self.family_call_kwargs["min_surfaces"], len(self.surface_configs)
        )
        # Strict inner0-based name postcondition ran exactly once, on the helper's band.
        self.assertEqual(len(self.strict_postcondition_calls), 1)
        self.assertIs(self.strict_postcondition_calls[0][0], _SENTINEL_SURFACE_DATA)
        self.assertEqual(
            self.strict_postcondition_calls[0][1],
            {"require_nesting": True},
        )
        # The standalone G-consistency check is NOT called directly on the full path
        # (it lives inside the strict postcondition).
        self.assertEqual(self.g_consistency_calls, [])
        # Caller returns the helper's band unchanged plus the empty warm-start list.
        self.assertIs(ordered_surface_data, _SENTINEL_SURFACE_DATA)
        self.assertEqual(warm_start_paths, [])

    def test_truncated_band_skips_strict_and_enforces_g_consistency(self):
        """Truncation ON + provenance.truncated -> strict SKIPPED, G-consistency RUNS."""
        self._install_family_stub(truncated=True)

        ordered_surface_data, warm_start_paths = self._call(
            allow_truncation=True, min_surfaces=2
        )

        # Caller forwards the requested truncation policy to the family helper.
        self.assertTrue(self.family_call_kwargs["allow_truncation"])
        self.assertEqual(self.family_call_kwargs["min_surfaces"], 2)
        # The strict inner0-based name postcondition must NOT run for a truncated
        # outer-suffix band (the helper already validated it with the relaxed check)...
        self.assertEqual(self.strict_postcondition_calls, [])
        # ...but the name-agnostic G-consistency check MUST still run, on that band.
        self.assertEqual(len(self.g_consistency_calls), 1)
        self.assertIs(self.g_consistency_calls[0], _SENTINEL_SURFACE_DATA)
        self.assertIs(ordered_surface_data, _SENTINEL_SURFACE_DATA)
        self.assertEqual(warm_start_paths, [])

    def test_truncation_on_but_full_acceptance_runs_strict_postcondition(self):
        """Truncation ON but provenance.truncated=False -> full strict postcondition RUNS."""
        self._install_family_stub(truncated=False)

        self._call(allow_truncation=True, min_surfaces=2)

        # allow_truncation was on, but the full requested stack was accepted
        # (truncated=False) so the band is inner0-based: strict runs once, and the
        # standalone G-consistency check is not invoked directly.
        self.assertTrue(self.family_call_kwargs["allow_truncation"])
        self.assertEqual(len(self.strict_postcondition_calls), 1)
        self.assertIs(self.strict_postcondition_calls[0][0], _SENTINEL_SURFACE_DATA)
        self.assertEqual(
            self.strict_postcondition_calls[0][1],
            {"require_nesting": True},
        )
        self.assertEqual(self.g_consistency_calls, [])

    def test_relaxed_initial_nesting_skips_only_the_setup_nesting_gate(self):
        """Opt-in relaxed admission preserves strict name/volume/G checks."""
        self._install_family_stub(truncated=False)

        ordered_surface_data, warm_start_paths = self._call(
            relax_initial_nesting=True,
        )

        self.assertEqual(len(self.strict_postcondition_calls), 1)
        self.assertIs(self.strict_postcondition_calls[0][0], _SENTINEL_SURFACE_DATA)
        self.assertEqual(
            self.strict_postcondition_calls[0][1],
            {"require_nesting": False},
        )
        self.assertEqual(self.g_consistency_calls, [])
        self.assertIs(ordered_surface_data, _SENTINEL_SURFACE_DATA)
        self.assertEqual(warm_start_paths, [])

    def test_contract_dispatch_forwards_truncation_policy_to_published_path(self):
        """initialize_surface_data_for_contract threads the policy into the published
        builder -- the propagation hop whose absence made the flag a silent no-op."""
        self._install_family_stub(truncated=False)
        module = self.module
        contract = SimpleNamespace(mode=module.PUBLISHED_MULTISURFACE)

        module.initialize_surface_data_for_contract(
            self.surface_configs,
            surface_mode_contract=contract,
            mpol=4,
            ntor=4,
            bs=object(),
            constraint_weight=1.0,
            default_iota=0.3,
            default_G=1.0,
            boozer_I=0.0,
            nfp=5,
            stage2_seed_surface=self.stage2_seed_surface,
            warm_start_surface_stem=None,
            allow_truncation=True,
            min_surfaces=2,
            relax_initial_nesting=True,
        )

        # The policy reached build_boozer_surface_family through the dispatcher.
        self.assertTrue(self.family_call_kwargs["allow_truncation"])
        self.assertEqual(self.family_call_kwargs["min_surfaces"], 2)
        self.assertEqual(
            self.strict_postcondition_calls[0][1],
            {"require_nesting": False},
        )


if __name__ == "__main__":
    unittest.main()
