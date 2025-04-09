import unittest
import pyrheed
from pyrheed.io import load_data
import matplotlib.pyplot as plt
import numpy as np

class TestPyrheed(unittest.TestCase):

    def setUp(self):
        # Load the data using the plugin
        self.r = load_data("docs/notebooks/example_data/si_111_7x7_112_t2.raw", plugin="umcs_arpes_raw")


    def test_set_center(self):
        # Test the set_center method
        try:
            self.r.R.set_center()
        except Exception as e:
            self.fail(f"set_center method raised an exception: {e}")


    def test_plot_image(self):
        # Test the plot_image method
        try:
            self.r.R.plot_image()
        except Exception as e:
            self.fail(f"plot_image method raised an exception: {e}")

if __name__ == '__main__': 
    unittest.main()