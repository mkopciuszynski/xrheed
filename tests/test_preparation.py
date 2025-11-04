import unittest
from pathlib import Path
import xrheed
from xrheed.preparation.alignment import find_horizontal_center, find_vertical_center
from xrheed.preparation.alignment import _spot_sigma_from_profile

# Expected centers for each dataset (cx, cy)
DATAFILE_CENTER_MAP = {
    "Si_111_7x7_112_phi_00.raw": (-0.27, -0.43),
    "Si_111_r3Ag_112_thA.raw": (-5.58, -4.24),
    "Si_111_r3Ag_112_thD.raw": (-5.53, 3.33),
    "Si_111_r3Ag_thA_phi15.raw": (-5.48, 1.83),
}


class TestCenterFinding(unittest.TestCase):
    """Integration tests for horizontal/vertical center detection on RHEED images."""

    def setUp(self):
        self.base = Path(__file__).parent / "data"

    def _load_image(self, fname: str):
        img = xrheed.load_data(self.base / fname, plugin="dsnp_arpes_raw")
        img.ri.screen_roi_width = 40
        img.ri.screen_roi_height = 60
        return img
    

    def test_spot_sigma_from_profile(self):
        fname = "Si_111_7x7_112_phi_00.raw"
        rheed_img = self._load_image(fname)
        
        profile = rheed_img.mean("sy")
        sigma = _spot_sigma_from_profile(profile)
        self.assertAlmostEqual(sigma, 0.41, places=1)

        profile = rheed_img.mean("sx")
        sigma = _spot_sigma_from_profile(profile)
        self.assertAlmostEqual(sigma, 0.37, places=1)

        profile = rheed_img.sel(sx=slice(-20,-10)).mean("sy")
        sigma = _spot_sigma_from_profile(profile)
        self.assertAlmostEqual(sigma, 0.43, places=1)

        profile = rheed_img.sel(sx=slice(-1,1)).mean("sx")
        sigma = _spot_sigma_from_profile(profile)
        self.assertAlmostEqual(sigma, 0.25, places=1)

        profile = profile * 0.0
        with self.assertWarns(UserWarning):
            sigma = _spot_sigma_from_profile(profile, max_sigma=5.0)

        self.assertAlmostEqual(sigma, 5.0, places=2)



    def test_center_detection(self):
        """Test initial center detection across multiple datasets."""
        for fname, (exp_cx, exp_cy) in DATAFILE_CENTER_MAP.items():
            with self.subTest(fname=fname):
                img = self._load_image(fname)
                roi = img.ri.get_roi_image()

                cx = find_horizontal_center(roi)
                cy = find_vertical_center(roi, center_x=cx)

                self.assertAlmostEqual(
                    cx, exp_cx, places=2,
                    msg=f"[{fname}] Initial cx mismatch: got {cx:.3f}, expected {exp_cx:.3f}"
                )
                self.assertAlmostEqual(
                    cy, exp_cy, places=2,
                    msg=f"[{fname}] Initial cy mismatch: got {cy:.3f}, expected {exp_cy:.3f}"
                )

    def test_manual_center_shift_recovery(self):
        """Test that manual center shifts are correctly recovered for the first dataset."""
        fname = "Si_111_7x7_112_phi_00.raw"
        img = self._load_image(fname)

        for dx, dy in [(-2.0, 0.0), (0.0, 2.0), (2.0, -2.0), (-2.0, 2.0)]:
            img.ri.set_center_auto()
            img.ri.set_center_manual(center_x=dx, center_y=dy)
            
            roi = img.ri.get_roi_image()
            cx = find_horizontal_center(roi)
            cy = find_vertical_center(roi,center_x=cx)

            self.assertAlmostEqual(
                cx, -dx, places=1,
                msg=f"[{fname}] cx mismatch for shift ({dx},{dy}): got {cx:.3f}, expected {dx:.3f}"
            )
            self.assertAlmostEqual(
                cy, -dy, places=1,
                msg=f"[{fname}] cy mismatch for shift ({dx},{dy}): got {cy:.3f}, expected {dy:.3f}"
            )


if __name__ == "__main__":
    unittest.main()
