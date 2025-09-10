import unittest
from pathlib import Path

import numpy as np
import xarray as xr
from xrheed.conversion import base, image
from xrheed.io import load_data


class TestBaseConversion(unittest.TestCase):
    def setUp(self):
        test_data_path = Path(__file__).parent / "data" / "Si_111_7x7_112_phi_00.raw"
        self.rheed_image = load_data(test_data_path, plugin="dsnp_arpes_raw")

    def test_convert_sx_to_ky(self):
        sx_coords_mm = np.array([0, 10, 20])
        ewald_sphere_radius = 2.0
        screen_sample_distance_mm = 10.0
        ky = base.convert_sx_to_ky(
            sx_coords_mm, ewald_sphere_radius, screen_sample_distance_mm
        )
        expected = np.array([0, 2, 4])
        np.testing.assert_array_almost_equal(ky, expected)

    def test_transform_to_kxky_shape(self):
        # Create a dummy RHEED image with required attributes
        rheed_image = self.rheed_image
        # Add required attributes for the function
        result = image.transform_image_to_kxky(rheed_image)
        self.assertIsInstance(result, xr.DataArray)
        self.assertTrue(result.shape[0] > 0 and result.shape[1] > 0)


if __name__ == "__main__":
    unittest.main()
