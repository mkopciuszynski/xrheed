import unittest

import numpy as np
from xrheed.kinematics.lattice import Lattice, rotation_matrix


class TestLattice(unittest.TestCase):
    def test_initialization(self):
        a1 = [1.0, 0.0]
        a2 = [0.0, 1.0]
        lattice = Lattice(a1, a2)
        np.testing.assert_array_equal(lattice.a1, np.array([1.0, 0.0, 0.0]))
        np.testing.assert_array_equal(lattice.a2, np.array([0.0, 1.0, 0.0]))

    def test_from_bulk_cubic(self):
        a = 5.43
        a_surf = a / np.sqrt(2)
        lattice = Lattice.from_bulk_cubic(a=a, cubic_type="FCC", plane="111")
        np.testing.assert_array_almost_equal(
            lattice.a1, np.array([0.0, a_surf, 0.0]), decimal=3
        )
        np.testing.assert_array_almost_equal(
            lattice.a2,
            np.array([a_surf * np.sqrt(3) * 0.5, a_surf * 0.5, 0.0]),
            decimal=3,
        )

        lattice = Lattice.from_bulk_cubic(a=a, cubic_type="SC", plane="110")
        np.testing.assert_array_almost_equal(
            lattice.a1, np.array([a * np.sqrt(2), 0.0, 0.0]), decimal=3
        )
        np.testing.assert_array_almost_equal(lattice.a2, np.array([0, a, 0]), decimal=3)

    def test_hex_lattice(self):
        a = 3.84
        a1, a2 = Lattice.hex_lattice(a)
        np.testing.assert_array_almost_equal(a1, np.array([0.0, a, 0.0]))
        np.testing.assert_array_almost_equal(
            a2, np.array([a * np.sqrt(3) * 0.5, a * 0.5, 0.0]), decimal=3
        )

    def test_rotate(self):
        a1 = [1.0, 0.0]
        a2 = [0.0, 1.0]
        lattice = Lattice(a1, a2)
        lattice.rotate(90)
        np.testing.assert_array_almost_equal(
            lattice.a1, np.array([0.0, -1.0, 0.0]), decimal=4
        )
        np.testing.assert_array_almost_equal(
            lattice.a2, np.array([1.0, 0.0, 0.0]), decimal=4
        )
        lattice.rotate(-90)
        np.testing.assert_array_almost_equal(
            lattice.a1, np.array([1.0, 0.0, 0.0]), decimal=4
        )
        np.testing.assert_array_almost_equal(
            lattice.a2, np.array([0.0, 1.0, 0.0]), decimal=4
        )

    def test_scale(self):
        a1 = [1.0, 0.0]
        a2 = [0.0, 1.0]
        lattice = Lattice(a1, a2)
        lattice.scale(2.0)
        np.testing.assert_array_almost_equal(lattice.a1, np.array([2.0, 0.0, 0.0]))
        np.testing.assert_array_almost_equal(lattice.a2, np.array([0.0, 2.0, 0.0]))

    def test_copy_and_deepcopy(self):
        a1 = [1.0, 0.0]
        a2 = [0.0, 1.0]
        lattice = Lattice(a1, a2)
        lattice_copy = lattice.__copy__()
        lattice_deepcopy = lattice.__deepcopy__({})
        np.testing.assert_array_equal(lattice.a1, lattice_copy.a1)
        np.testing.assert_array_equal(lattice.a2, lattice_deepcopy.a2)
        # Ensure they are not the same object
        self.assertIsNot(lattice, lattice_copy)
        self.assertIsNot(lattice, lattice_deepcopy)

    def test_str(self):
        a1 = [1.0, 0.0]
        a2 = [0.0, 1.0]
        lattice = Lattice(a1, a2)
        s = str(lattice)
        self.assertIn("a1", s)
        self.assertIn("a2", s)

    def test_invalid_vector(self):
        with self.assertRaises(ValueError):
            Lattice([1], [0, 1])
        with self.assertRaises(ValueError):
            Lattice([1, 0, 0, 0], [0, 1])

    def test_plot_methods(self):
        # Just check that plotting does not raise exceptions
        a1 = [1.0, 0.0]
        a2 = [0.0, 1.0]
        lattice = Lattice(a1, a2)
        lattice.plot_real()
        lattice.plot_reciprocal()

    def test_generate_lattice(self):
        a1 = np.array([3.84, 0, 0])
        a2 = np.array([1.92, 3.325, 0])
        points = Lattice.generate_lattice(a1, a2, space_size=10.0)
        assert points.ndim == 2
        assert points.shape[1] == 3

    def test_repr(self):
        a1 = [3.84, 0]
        a2 = [1.92, 3.325]
        lattice = Lattice(a1, a2)
        s = repr(lattice)
        assert "a1" in s and "a2" in s

    def test_rotation_matrix(self):
        mat = rotation_matrix(90)
        assert mat.shape == (3, 3)
        # 90 degree rotation should swap x and y
        v = np.array([1, 0, 0], dtype=np.float32)
        v_rot = mat @ v
        assert np.allclose(v_rot[:2], [0, -1], atol=0.001)


if __name__ == "__main__":
    unittest.main()
