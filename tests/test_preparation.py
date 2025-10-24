import unittest
from pathlib import Path
import numpy as np
import xrheed
from xrheed.preparation.alignment import find_horizontal_center, find_vertical_center

# Define expected centers for each dataset
DATAFILE_CENTER_MAP = {
    "Si_111_7x7_112_phi_00.raw": (-0.44, 1.64),
    "Si_111_r3Ag_112_thA.raw": (-5.78, 1.10),
    "Si_111_r3Ag_112_thD.raw": (-5.97, 1.10),
    "Si_111_r3Ag_thA_phi15.raw": (-5.97, 1.10),
}

class TestCenterFinding(unittest.TestCase):
    def setUp(self):
        self.base = Path(__file__).parent / "data"

    def test_center_detection_and_shifts(self):
        for fname, (exp_cx, exp_cy) in DATAFILE_CENTER_MAP.items():
            with self.subTest(fname=fname):
                rheed_image = xrheed.load_data(
                    self.base / fname, plugin="dsnp_arpes_raw"
                )

                # --- Step 1: check initial center values directly after loading ---
                cx = find_horizontal_center(rheed_image)
                cy = find_vertical_center(rheed_image, shadow_edge_width=5.0)

                if exp_cx is not None and exp_cy is not None:
                    self.assertAlmostEqual(
                        cx,
                        exp_cx,
                        places=2,
                        msg=f"[{fname}] Initial cx mismatch: got {cx:.3f}, expected {exp_cx:.3f}",
                    )
                    self.assertAlmostEqual(
                        cy,
                        exp_cy,
                        places=2,
                        msg=f"[{fname}] Initial cy mismatch: got {cy:.3f}, expected {exp_cy:.3f}",
                    )

                # --- Step 2: auto-center the image ---
                rheed_image.ri.set_center_auto()
                cx_auto = find_horizontal_center(rheed_image)
                cy_auto = find_vertical_center(rheed_image)

                self.assertAlmostEqual(
                    cx_auto,
                    0.0,
                    places=2,
                    msg=f"[{fname}] Auto-centered cx not ~0: got {cx_auto:.3f}",
                )
                self.assertAlmostEqual(
                    cy_auto,
                    0.0,
                    places=2,
                    msg=f"[{fname}] Auto-centered cy not ~0: got {cy_auto:.3f}",
                )

                # --- Step 3: apply manual shifts and verify recovery ---
                for dx, dy in [(-0.5, 0.5), (1.0, -1.0), (-1.5, 0.2)]:
                    rheed_image.ri.set_center_manual(center_x=dx, center_y=dy)

                    cx_shift = find_horizontal_center(rheed_image)
                    cy_shift = find_vertical_center(rheed_image)

                    self.assertAlmostEqual(
                        cx_shift,
                        dx,
                        places=2,
                        msg=f"[{fname}] cx mismatch for shift ({dx},{dy}): got {cx_shift:.3f}, expected {dx:.3f}",
                    )
                    self.assertAlmostEqual(
                        cy_shift,
                        dy,
                        places=2,
                        msg=f"[{fname}] cy mismatch for shift ({dx},{dy}): got {cy_shift:.3f}, expected {dy:.3f}",
                    )


if __name__ == "__main__":
    unittest.main()
