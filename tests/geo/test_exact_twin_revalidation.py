"""Coverage for the exact-twin per-eval Boozer re-solve handoff and its post-handoff
re-validation.

Two concerns, previously untested (commit f84008ce shipped the feature with no tests;
the post-handoff re-validation was added as a watchdog review-fix):

  1. ``_resolve_published_family_to_exact_twins`` rebinds each entry's ``boozer_surface``
     to its exact-Newton twin on success, and FAILS CLOSED (RuntimeError) if any twin does
     not re-solve from its converged least-squares dofs.

  2. ``initialize_published_surface_data_from_stage2_seed`` in
     ``MULTISURFACE_RESOLVE_MODE='exact_twin'`` RE-VALIDATES the twin-replaced stack after
     the in-place handoff (name-agnostic volume-order + nesting + shared-G). The
     published-family postconditions run on the least-squares family BEFORE the handoff, and
     exact Newton converges to the NEAREST Boozer surface, so a twin can drift its volume,
     geometry, or G; without the post-handoff re-check a volume-crossed, non-nested, or
     G-inconsistent twin stack would silently seed the optimizer warm-start. These tests prove
     the re-check rejects such a stack and is inert under the default ``ls`` mode.

Everything is stubbed -- no Boozer solve or geometry runs.
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


def _fake_boozer_surface(*, volume, G, iota=0.3, nested=True):
    """Stand-in BoozerSurface exposing only what the resolver + re-validation read:
    ``.surface.volume()`` (``_require_published_volume_order``) and ``.res['G']`` /
    ``.res['iota']`` (``_boozer_res_scalar`` / the resolver warm-start)."""
    return SimpleNamespace(
        surface=SimpleNamespace(volume=lambda v=volume: v, nested=bool(nested)),
        res={"G": G, "iota": iota},
    )


class ExactTwinResolverTests(unittest.TestCase):
    """``_resolve_published_family_to_exact_twins`` rebind + fail-closed semantics."""

    @classmethod
    def setUpClass(cls):
        cls.module = _load_single_stage_example_module()

    def _patch(self, name, value):
        original = getattr(self.module, name)
        setattr(self.module, name, value)
        self.addCleanup(setattr, self.module, name, original)

    def _stack(self):
        return [
            {"name": "inner0", "target_volume": 1.0,
             "boozer_surface": _fake_boozer_surface(volume=1.0, G=2.0)},
            {"name": "outer", "target_volume": 2.0,
             "boozer_surface": _fake_boozer_surface(volume=2.0, G=2.0)},
        ]

    def test_resolve_rebinds_each_boozer_surface_to_its_twin_on_success(self):
        stack = self._stack()

        def fake_impl(ls_surface, mpol, ntor, bs, target_volume, constraint_weight,
                      iota, G, *, boozer_I, initial_surface_guess, nfp):
            # A distinct twin per surface, tagged by target_volume so the test can prove
            # the right twin landed on the right entry (not just "some object").
            return SimpleNamespace(success=True,
                                   boozer_surface=("twin", target_volume))

        self._patch("_attempt_initialize_boozer_surface_impl", fake_impl)

        self.module._resolve_published_family_to_exact_twins(
            stack, mpol=4, ntor=4, bs=object(), boozer_I=0.0, nfp=5)

        self.assertEqual([entry["boozer_surface"] for entry in stack],
                         [("twin", 1.0), ("twin", 2.0)])

    def test_resolve_fails_closed_when_a_twin_does_not_reconverge(self):
        stack = self._stack()
        original_surfaces = [entry["boozer_surface"] for entry in stack]

        def fake_impl(ls_surface, mpol, ntor, bs, target_volume, constraint_weight,
                      iota, G, *, boozer_I, initial_surface_guess, nfp):
            return SimpleNamespace(success=False, solve_success=False,
                                   self_intersecting=True, error_type="iota_collapse",
                                   boozer_surface=None)

        self._patch("_attempt_initialize_boozer_surface_impl", fake_impl)

        with self.assertRaises(RuntimeError) as ctx:
            self.module._resolve_published_family_to_exact_twins(
                stack, mpol=4, ntor=4, bs=object(), boozer_I=0.0, nfp=5)
        message = str(ctx.exception)
        # Fail-loud AND diagnostic: the failing surface name + the twin's error_type surface
        # in the message (a dropped/typo'd diagnostic field would otherwise pass silently).
        self.assertIn("exact-twin handoff failed", message)
        self.assertIn("inner0", message)          # the failing surface's name
        self.assertIn("iota_collapse", message)   # twin_result.error_type
        # Fail-closed: the first entry was never rebound to a None/garbage surface.
        self.assertIs(stack[0]["boozer_surface"], original_surfaces[0])


class ExactTwinRevalidationWiringTests(unittest.TestCase):
    """``initialize_published_surface_data_from_stage2_seed`` re-validates the twin stack.

    Mirrors the stubbing harness of ``test_surface_truncation_wiring`` (no solver/geometry
    runs): the family build is replaced by a stub that returns a controllable band, and the
    exact-twin handoff is stubbed to set each twin's solved volume/nesting/G -- so a test drives
    the REAL post-handoff ``_require_published_volume_order`` /
    ``_require_published_surface_nesting`` / ``_require_published_G_consistency`` invariants on a
    chosen twin outcome. The pre-handoff strict postcondition is stubbed inert (the least-squares
    band is valid by construction); we are testing the POST-handoff re-check.
    """

    @classmethod
    def setUpClass(cls):
        cls.module = _load_single_stage_example_module()

    def setUp(self):
        self.surface_configs = [
            {"name": "inner0", "target_volume": 1.0},
            {"name": "outer", "target_volume": 2.0},
        ]
        # The band the family build "returns"; the twin handoff mutates it in place and the
        # caller hands it straight back, so tests assert identity on the return.
        self.family_band = [
            {"name": "inner0", "target_volume": 1.0},
            {"name": "outer", "target_volume": 2.0},
        ]
        self._patch("_require_published_stage2_solved_seed", lambda seed: None)
        self._patch("_published_surface_names",
                    lambda count: ["inner0", "outer"][:count])
        self._patch("BoozerFamilyOuterSeed",
                    lambda *, surface, iota, G: SimpleNamespace(surface=surface, iota=iota, G=G))
        self._patch("_require_finite_explicit_G", lambda name, G: G)
        self._patch("contract_surface_to_target_volume",
                    lambda previous_surface, target_volume: previous_surface)
        self._patch("_surface_data_entry",
                    lambda config, boozer_surface, provenance: {"name": config["name"]})
        # Pre-handoff strict postcondition: inert. The LS band is valid by construction; the
        # behaviour under test is the POST-handoff re-validation, which must run for real.
        self._patch(
            "_require_published_surface_data_postconditions",
            lambda data, **kwargs: None,
        )
        self._patch(
            "cross_sections_are_nested",
            lambda inner, outer: (bool(getattr(inner, "nested", True)), []),
        )
        self._patch("build_boozer_surface_family", self._fake_family_build)
        self._set_resolve_mode("exact_twin")
        self.stage2_seed_surface = SimpleNamespace(surface=object(), iota=0.3, G=1.0)

    def _patch(self, name, value):
        original = getattr(self.module, name)
        setattr(self.module, name, value)
        self.addCleanup(setattr, self.module, name, original)

    def _set_resolve_mode(self, mode):
        module = self.module
        had = hasattr(module, "MULTISURFACE_RESOLVE_MODE")
        old = getattr(module, "MULTISURFACE_RESOLVE_MODE", None)
        module.MULTISURFACE_RESOLVE_MODE = mode

        def restore():
            if had:
                module.MULTISURFACE_RESOLVE_MODE = old
            else:
                delattr(module, "MULTISURFACE_RESOLVE_MODE")

        self.addCleanup(restore)

    def _fake_family_build(self, ordered_configs, **kwargs):
        return self.family_band, SimpleNamespace(truncated=False)

    def _patch_twin_outcome(self, *, volumes, Gs, nested=None):
        """Stub the handoff to set each twin's solved volume/nesting/G."""
        nested_flags = [True] * len(volumes) if nested is None else list(nested)

        def fake_resolve(ordered_surface_data, **kwargs):
            for entry, volume, G, is_nested in zip(
                ordered_surface_data, volumes, Gs, nested_flags
            ):
                entry["boozer_surface"] = _fake_boozer_surface(
                    volume=volume, G=G, nested=is_nested
                )
        self._patch("_resolve_published_family_to_exact_twins", fake_resolve)

    def _call(self):
        return self.module.initialize_published_surface_data_from_stage2_seed(
            self.surface_configs,
            mpol=4, ntor=4, bs=object(), constraint_weight=1.0,
            boozer_I=0.0, nfp=5,
            stage2_seed_surface=self.stage2_seed_surface,
            allow_truncation=False, min_surfaces=2,
        )

    def test_revalidation_rejects_twin_volume_crossing(self):
        """Twins drift so volumes are no longer strictly inner-to-outer increasing -> raise."""
        self._patch_twin_outcome(volumes=[2.0, 1.0], Gs=[2.0, 2.0])
        with self.assertRaisesRegex(RuntimeError, "volumes must be strictly ordered"):
            self._call()

    def test_revalidation_rejects_twin_G_drift(self):
        """A twin's solved G drifts off the shared outer G -> raise."""
        self._patch_twin_outcome(volumes=[1.0, 2.0], Gs=[2.0, 2.5])
        with self.assertRaisesRegex(RuntimeError, "solved G drifted"):
            self._call()

    def test_revalidation_rejects_twin_non_nesting(self):
        """Volume-ordered twins that are geometrically non-nested fail closed."""
        self._patch_twin_outcome(
            volumes=[1.0, 2.0],
            Gs=[2.0, 2.0],
            nested=[False, True],
        )
        with self.assertRaisesRegex(RuntimeError, "strictly nested"):
            self._call()

    def test_revalidation_passes_for_consistent_nested_twins(self):
        """Nested (strictly increasing volume), shared-G twins pass and return the band."""
        self._patch_twin_outcome(volumes=[1.0, 2.0], Gs=[2.0, 2.0])
        ordered_surface_data, warm_start_paths = self._call()
        self.assertIs(ordered_surface_data, self.family_band)
        self.assertEqual(warm_start_paths, [])

    def test_ls_default_mode_neither_resolves_nor_revalidates_twins(self):
        """Default 'ls' mode skips the twin block entirely (byte-unchanged legacy path)."""
        self.module.MULTISURFACE_RESOLVE_MODE = "ls"  # setUp's cleanup restores it
        resolve_calls = []
        self._patch("_resolve_published_family_to_exact_twins",
                    lambda *a, **k: resolve_calls.append(True))

        def _must_not_run(data):
            raise AssertionError("post-handoff re-validation must not run in 'ls' mode")

        self._patch("_require_published_volume_order", _must_not_run)
        self._patch("_require_published_surface_nesting", _must_not_run)
        self._patch("_require_published_G_consistency", _must_not_run)

        ordered_surface_data, warm_start_paths = self._call()
        self.assertEqual(resolve_calls, [])
        self.assertIs(ordered_surface_data, self.family_band)


if __name__ == "__main__":
    unittest.main()
