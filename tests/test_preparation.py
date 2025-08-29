import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from xrheed.io import load_data
from xrheed.preparation.alignment import find_horizontal_center, find_vertical_center


class TestDataLoading(unittest.TestCase):
    def setUp(self):
        test_data_path = Path(__file__).parent / "data" / "Si_111_7x7_112_phi_00.raw"
        self.rheed_image = load_data(test_data_path, plugin="dsnp_arpes_raw")

    def test_set_center(self):
        center_x = -0.5
        center_y = 0.5

        self.rheed_image.ri.apply_image_center(center_x=center_x, center_y=center_y)

        center_x = find_horizontal_center(self.rheed_image)
        center_y = find_vertical_center(self.rheed_image, shadow_edge_width=5.0)

        # Correct assertion
        self.assertAlmostEqual(center_x, 0.28, places=2)
        self.assertAlmostEqual(center_y, 1.13, places=2)

    def test_plot_image(self):
        try:
            self.rheed_image.ri.plot_image()
        except Exception as e:
            self.fail(f"plot_image method raised an exception: {e}")


if __name__ == "__main__":
    unittest.main()
