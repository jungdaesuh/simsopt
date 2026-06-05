import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = REPO_ROOT / "examples" / "single_stage_optimization"
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

from banana_opt.vmec_curtor_sign import (  # noqa: E402
    build_curtor_sign_artifact,
    radial_schedule_for_final_ns,
)


def _record(curtor_A: float, iota_edge: float, *, ns: int = 51) -> dict[str, object]:
    return {
        "curtor_A": curtor_A,
        "vmec_final_ns": ns,
        "iota_axis": 0.4 + 0.25 * (iota_edge - 0.3),
        "iota_edge": iota_edge,
        "converged": True,
    }


class VmecCurtorSignTests(unittest.TestCase):
    def test_radial_schedule_uses_requested_final_ns(self):
        self.assertEqual(radial_schedule_for_final_ns(51), (13, 25, 51))
        self.assertEqual(radial_schedule_for_final_ns(99), (13, 25, 51, 99))
        self.assertEqual(
            radial_schedule_for_final_ns(151),
            (13, 25, 51, 99, 151),
        )

    def test_curtor_sign_artifact_selects_edge_lift_branch(self):
        artifact = build_curtor_sign_artifact(
            sign_records=[
                _record(0.0, 0.3000),
                _record(-800.0, 0.2754),
                _record(800.0, 0.3242),
            ],
            convergence_records=[
                _record(0.0, 0.3000, ns=51),
                _record(800.0, 0.3242, ns=51),
                _record(0.0, 0.3010, ns=99),
                _record(800.0, 0.3250, ns=99),
                _record(0.0, 0.3012, ns=151),
                _record(800.0, 0.3251, ns=151),
            ],
            source_input="/tmp/input.s01",
            current_magnitude_A=800.0,
        )

        self.assertEqual(
            artifact["sign_probe"]["iota_raising_curtor_sign"],
            1,
        )
        self.assertAlmostEqual(
            artifact["sign_probe"]["validated_edge_lift"],
            0.0242,
        )
        edge_lifts = artifact["convergence"]["edge_lifts"]
        self.assertEqual([row["ns"] for row in edge_lifts], [51, 99, 151])
        self.assertEqual(set(edge_lifts[0]), {"ns", "curtor_A", "edge_lift"})


if __name__ == "__main__":
    unittest.main()
