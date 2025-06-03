import unittest
from pathlib import Path

from xrheed.io import load_data

class TestDataLoading(unittest.TestCase):

    def setUp(self):
        # Load the data using the plugin
        test_data_path = Path(__file__).parent / "data" / "si_111_7x7_112.raw"
        self.rheed_image = load_data(str(test_data_path), plugin="umcs_arpes_raw")

    def test_set_center(self):
        # Test the set_center method
        try:
            self.rheed_image.R.set_center()
        except Exception as e:
            self.fail(f"set_center method raised an exception: {e}")

    def test_plot_image(self):
        # Test the plot_image method
        try:
            self.rheed_image.R.plot_image()
        except Exception as e:
            self.fail(f"plot_image method raised an exception: {e}")

if __name__ == '__main__': 
    unittest.main()