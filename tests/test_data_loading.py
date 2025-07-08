import unittest
from pathlib import Path
import matplotlib

matplotlib.use("Agg")

from xrheed.io import load_data


class TestDataLoading(unittest.TestCase):
    def setUp(self):
        test_data_path = Path(__file__).parent / "data" / "Si_111_7x7_112_phi_00.raw"
        self.rheed_image = load_data(test_data_path, plugin="dsnp_arpes_raw")

    def test_image_attributes(self):
        attrs = self.rheed_image.attrs

        # Check that each attribute is present and is a float
        required_attrs = ["screen_scale", "beam_energy", "screen_sample_distance"]
        for attr in required_attrs:
            self.assertIn(attr, attrs, msg=f"Missing attribute: {attr}")
            self.assertIsInstance(
                attrs[attr], (float, int), msg=f"{attr} is not a number"
            )


if __name__ == "__main__":
    unittest.main()
