import unittest
from pathlib import Path

import numpy as np
import xarray as xr
from xrheed.io import load_data
from xrheed.kinematics.ewald import Ewald
from xrheed.kinematics.lattice import Lattice


class TestEwald(unittest.TestCase):
    def setUp(self):
        # Prepare test lattice Si(111)-(1x1)
        self.lattice = Lattice.from_bulk_cubic(a=5.43, plane="111")

        # Prepare test image
        test_data_path = Path(__file__).parent / "data" / "Si_111_7x7_112_phi_00.raw"
        self.rheed_image = load_data(test_data_path, plugin="dsnp_arpes_raw")
        self.rheed_image.ri.apply_image_center(auto_center=True)
        self.rheed_image.ri.beta = 2.8

    def test_init_without_image(self):
        ewald = Ewald(self.lattice, image=None)
        self.assertFalse(ewald._image_data_available)
        self.assertIsInstance(ewald.beam_energy, float)

    def test_init_with_image(self):
        ewald = Ewald(self.lattice, image=self.rheed_image)
        self.assertTrue(ewald._image_data_available)
        self.assertIsInstance(ewald.image, xr.DataArray)

    def test_calculate_ewald(self):
        ewald = Ewald(self.lattice, self.rheed_image)
        ewald.calculate_ewald()

        self.assertTrue(hasattr(ewald, "ew_sx"))
        self.assertTrue(hasattr(ewald, "ew_sy"))
        self.assertGreater(len(ewald.ew_sx), 0)
        self.assertGreater(len(ewald.ew_sy), 0)

    def test_property_updates_trigger_calculation(self):
        ewald = Ewald(self.lattice, self.rheed_image)
        ewald.calculate_ewald()

        old_sx = ewald.ew_sx.copy()
        ewald.alpha = ewald.alpha + 5.0
        self.assertFalse(np.array_equal(old_sx, ewald.ew_sx))

    def test_inverse_lattice_generation(self):
        ewald = Ewald(self.lattice, self.rheed_image)
        inv_lattice = ewald._prepare_inverse_lattice()
        self.assertIsInstance(inv_lattice, np.ndarray)
        # self.assertEqual(inv_lattice.shape[1], 2)

    def test_generate_mask(self):
        ewald = Ewald(self.lattice, self.rheed_image)
        ewald.calculate_ewald()
        mask = ewald._generate_mask()

        self.assertEqual(mask.shape, ewald.image.shape)
        self.assertTrue(mask.dtype == bool)

    def test_match_calculation(self):
        ewald = Ewald(self.lattice, self.rheed_image)
        ewald.calculate_ewald()

        score_A = ewald.calculate_match()
        ewald.alpha = 10.0
        score_B = ewald.calculate_match()

        self.assertIsInstance(score_A, (int, np.integer))
        self.assertTrue(score_A > score_B)

    def test_repr_returns_string(self):
        ewald = Ewald(self.lattice, self.rheed_image)
        r = repr(ewald)
        self.assertIsInstance(r, str)
        self.assertIn("Ewald", r)


if __name__ == "__main__":
    unittest.main()
