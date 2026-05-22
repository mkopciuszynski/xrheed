import unittest
from pathlib import Path

import numpy as np
import xarray as xr

import xrheed
from xrheed.kinematics.ewald import Ewald
from xrheed.kinematics.lattice import Lattice


class TestEwald(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Prepare test lattice Si(111)-(1x1)
        cls.lattice = Lattice.from_bulk_cubic(a=5.43, plane="111")

        # Prepare test image
        test_data_path = Path(__file__).parent / "data" / "Si_111_7x7_112_phi_00.raw"

        cls.rheed_image = xrheed.load_data(
            test_data_path,
            plugin="dsnp_arpes_raw",
        )

        cls.rheed_image.ri.set_center_auto(update_incident_angle=True)

        cls.rheed_image.ri.alpha = 0.0

    def setUp(self):
        self.ewald = Ewald(
            self.lattice,
            self.rheed_image,
        )

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def test_init_without_image(self):
        ewald = Ewald(self.lattice, image=None)

        self.assertFalse(ewald._image_data_available)
        self.assertIsNone(ewald.image)

        self.assertIsInstance(ewald.beam_energy, float)
        self.assertAlmostEqual(ewald.beam_energy, 18_600.0)

    def test_init_with_image(self):
        self.assertTrue(self.ewald._image_data_available)

        self.assertIsInstance(
            self.ewald.image,
            xr.DataArray,
        )

    # ------------------------------------------------------------------
    # Geometry calculation
    # ------------------------------------------------------------------

    def test_calculate_ewald(self):
        self.ewald.calculate_ewald()

        self.assertTrue(hasattr(self.ewald, "ew_sx"))
        self.assertTrue(hasattr(self.ewald, "ew_sy"))

        self.assertGreater(len(self.ewald.ew_sx), 0)
        self.assertGreater(len(self.ewald.ew_sy), 0)

    def test_ewald_coordinate_shapes_match(self):
        self.assertEqual(
            self.ewald.ew_sx.shape,
            self.ewald.ew_sy.shape,
        )

    def test_ewald_coordinates_contain_no_nan(self):
        self.assertFalse(np.isnan(self.ewald.ew_sx).any())
        self.assertFalse(np.isnan(self.ewald.ew_sy).any())

    def test_inverse_lattice_generation(self):
        inv_lattice = self.ewald._prepare_inverse_lattice()

        self.assertIsInstance(inv_lattice, np.ndarray)
        self.assertGreaterEqual(inv_lattice.shape[1], 2)

    # ------------------------------------------------------------------
    # Mutable state updates
    # ------------------------------------------------------------------

    def test_azimuthal_rotation_updates_ewald_positions(self):
        old_sx = self.ewald.ew_sx.copy()

        self.ewald.ewald_azimuthal_rotation += 5.0

        self.assertFalse(np.array_equal(old_sx, self.ewald.ew_sx))

    def test_lattice_scale_updates_ewald_positions(self):
        old_sx = self.ewald.ew_sx.copy()

        self.ewald.lattice_scale = 1.2

        self.assertFalse(np.array_equal(old_sx, self.ewald.ew_sx))

    def test_incident_angle_updates_ewald_positions(self):
        old_sy = self.ewald.ew_sy.copy()

        self.ewald.incident_angle += 0.2

        self.assertFalse(np.array_equal(old_sy, self.ewald.ew_sy))

    def test_ewald_roi_updates_inverse_lattice(self):
        old_size = len(self.ewald._inverse_lattice)

        self.ewald.ewald_roi = 20

        new_size = len(self.ewald._inverse_lattice)

        self.assertNotEqual(old_size, new_size)

    # ------------------------------------------------------------------
    # Spot structure and mask generation
    # ------------------------------------------------------------------

    def test_spot_structure_exists(self):
        self.assertTrue(isinstance(self.ewald.spot_structure, np.ndarray))

        self.assertEqual(
            self.ewald.spot_structure.dtype,
            bool,
        )

    def test_set_spot_size_updates_structure(self):
        old_shape = self.ewald.spot_structure.shape

        self.ewald.set_spot_size(
            width=2.0,
            height=6.0,
        )

        new_shape = self.ewald.spot_structure.shape

        self.assertNotEqual(old_shape, new_shape)

    def test_generate_mask(self):
        mask = self.ewald._generate_mask()

        self.assertEqual(
            mask.shape,
            self.ewald.image.shape,
        )

        self.assertEqual(mask.dtype, bool)

    def test_generated_mask_contains_pixels(self):
        mask = self.ewald._generate_mask()

        self.assertGreater(
            np.count_nonzero(mask),
            0,
        )

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def test_calculate_match(self):
        score_A = self.ewald.calculate_match()

        self.ewald.ewald_azimuthal_rotation = 10.0

        score_B = self.ewald.calculate_match()

        self.assertIsInstance(
            score_A,
            (int, np.integer),
        )

        self.assertGreater(score_A, score_B)

    # ------------------------------------------------------------------
    # Stack handling
    # ------------------------------------------------------------------

    def test_stack_initialization(self):
        stack = xr.concat(
            [self.rheed_image, self.rheed_image],
            dim="frame",
        )

        ewald = Ewald(
            self.lattice,
            stack,
        )

        self.assertEqual(ewald.stack_index, 0)

        self.assertIsNotNone(ewald.image)

    def test_stack_index_switches_image(self):
        image_A = self.rheed_image.copy()

        image_B = self.rheed_image.copy() * 2

        stack = xr.concat(
            [image_A, image_B],
            dim="frame",
        )

        ewald = Ewald(
            self.lattice,
            stack,
        )

        first_image = ewald.image.copy()

        ewald.stack_index = 1

        self.assertEqual(ewald.stack_index, 1)

        self.assertFalse(
            np.array_equal(
                first_image.data,
                ewald.image.data,
            )
        )

    def test_invalid_stack_index_raises(self):
        stack = xr.concat(
            [self.rheed_image, self.rheed_image],
            dim="frame",
        )

        ewald = Ewald(
            self.lattice,
            stack,
        )

        with self.assertRaises(ValueError):
            ewald.stack_index = 100

    # ------------------------------------------------------------------
    # Public API regression checks
    # ------------------------------------------------------------------

    def test_public_matching_methods_exist(self):
        self.assertTrue(hasattr(self.ewald, "match_alpha"))

        self.assertTrue(hasattr(self.ewald, "match_scale"))

        self.assertTrue(hasattr(self.ewald, "match_alpha_scale"))

        self.assertTrue(hasattr(self.ewald, "calculate_match"))

    def test_plot_methods_exist(self):
        self.assertTrue(hasattr(self.ewald, "plot"))

        self.assertTrue(hasattr(self.ewald, "plot_spots"))


if __name__ == "__main__":
    unittest.main()
