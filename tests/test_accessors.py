import unittest
from pathlib import Path
import numpy as np
import xarray as xr
import xrheed


class TestRHEEDAccessors(unittest.TestCase):
    def setUp(self):
        file_path = Path(__file__).parent / "data" / "Test_image_UMCS_DSNP_ARPES.BMP"
        self.da = xrheed.load_data(
            file_path,
            screen_sample_distance=309.2,
            screen_scale=9.04,
            beam_energy=18_600,
        )

    def test_rotate(self):
        da_copy = self.da.copy()
        da_copy.ri.rotate(2.0)  # rotate in place
        self.assertEqual(da_copy.shape, self.da.shape)  # shape must be preserved

    def test_set_center_manual(self):
        da_copy = self.da.copy()
        
        original_sx = da_copy.sx.values.copy()
        original_sy = da_copy.sy.values.copy()
        
        da_copy.ri.set_center_manual(center_x=5.0, center_y=3.0)

        self.assertTrue(np.allclose(da_copy.sx.values, original_sx - 5.0))
        self.assertTrue(np.allclose(da_copy.sy.values, original_sy - 3.0))

    def test_get_profile_default(self):
        profile = self.da.ri.get_profile(center=(0.0, 0.0), width=20.0, height=20.0)
        self.assertIsInstance(profile, xr.DataArray)
        self.assertIn("sx", profile.dims)  # reduced over sy
        self.assertNotIn("sy", profile.dims)
        self.assertIn("profile_center", profile.attrs)

    def test_plot_image_runs(self):
        ax = self.da.ri.plot_image()
        self.assertIsNotNone(ax)

    def test_profile_accessor_convert_to_k(self):
        profile = self.da.ri.get_profile(center=(0, 0), width=20, height=20)
        ky_profile = profile.rp.convert_to_k()
        self.assertIn("ky", ky_profile.dims)


if __name__ == "__main__":
    unittest.main()
