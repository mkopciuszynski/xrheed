import unittest
from pathlib import Path
import xrheed
from xrheed.preparation.alignment import find_horizontal_center, find_vertical_center

# Expected centers for each dataset (cx, cy)
DATAFILE_CENTER_MAP = {
    "Si_111_7x7_112_phi_00.raw": (-0.2950, -0.4425),
    "Si_111_r3Ag_112_thA.raw": (-5.6969, -4.3695),
    "Si_111_r3Ag_112_thD.raw": (-5.7107, 3.4292),
    "Si_111_r3Ag_thA_phi15.raw": (-5.6416, 1.8805),
}


class TestCenterFinding(unittest.TestCase):
    """Integration tests for horizontal/vertical center detection on RHEED images."""

    def setUp(self):
        self.base = Path(__file__).parent / "data"

    def _load_roi(self, fname: str):
        """Load a RHEED image and return its ROI image."""
        img = xrheed.load_data(self.base / fname, plugin="dsnp_arpes_raw")
        img.ri.screen_roi_width = 40
        img.ri.screen_roi_height = 60
        return img, img.ri.get_roi_image()

    def _assert_initial_center(self, fname: str, roi, exp_cx: float, exp_cy: float):
        """Check that the initial detected centers match expected values."""
        cx = find_horizontal_center(roi)
        cy = find_vertical_center(roi, center_x=cx)

        self.assertAlmostEqual(
            cx,
            exp_cx,
            places=2,
            msg=f"[{fname}] Initial cx mismatch: got {cx:.2f}, expected {exp_cx:.2f}",
        )
        self.assertAlmostEqual(
            cy,
            exp_cy,
            places=2,
            msg=f"[{fname}] Initial cy mismatch: got {cy:.2f}, expected {exp_cy:.2f}",
        )
        return cx, cy

    def _assert_auto_center(self, fname: str, img, cx: float, cy: float):
        """Check that after manual centering, the detected centers are ~0."""
        img.ri.set_center_manual(cx, cy)
        roi = img.ri.get_roi_image()
        cx_auto = find_horizontal_center(roi)
        cy_auto = find_vertical_center(roi)

        self.assertAlmostEqual(
            cx_auto,
            0.0,
            places=1,
            msg=f"[{fname}] Auto-centered cx not ~0: got {cx_auto:.2f}",
        )
        self.assertAlmostEqual(
            cy_auto,
            0.0,
            places=1,
            msg=f"[{fname}] Auto-centered cy not ~0: got {cy_auto:.2f}",
        )

    def test_center_detection(self):
        """Test center detection and auto-centering across multiple datasets."""
        for fname, (exp_cx, exp_cy) in DATAFILE_CENTER_MAP.items():
            with self.subTest(fname=fname):
                img, roi = self._load_roi(fname)
                cx, cy = self._assert_initial_center(fname, roi, exp_cx, exp_cy)
                self._assert_auto_center(fname, img, cx, cy)


if __name__ == "__main__":
    unittest.main()
