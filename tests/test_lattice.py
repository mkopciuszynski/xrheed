import unittest
import numpy as np
import copy
from xrheed.kinematics.lattice import Lattice

class TestLattice(unittest.TestCase):

    def test_initialization(self):
        a1 = [1.0, 0.0]
        a2 = [0.0, 1.0]
        lattice = Lattice(a1, a2)
        np.testing.assert_array_equal(lattice.a1, np.array([1.0, 0.0, 0.0]))
        np.testing.assert_array_equal(lattice.a2, np.array([0.0, 1.0, 0.0]))

    def test_from_bulk_cubic(self):
        a = 5.43
        lattice = Lattice.from_bulk_cubic(a=a, cubic_type="FCC", plane="111")
        np.testing.assert_array_almost_equal(lattice.a1, np.array([3.84, 0.0, 0.0]), decimal=3)
        np.testing.assert_array_almost_equal(lattice.a2, np.array([3.84*0.5, 3.325, 0]), decimal=3)

        lattice = Lattice.from_bulk_cubic(a=a, cubic_type="SC", plane="110")
        np.testing.assert_array_almost_equal(lattice.a1, np.array([a * np.sqrt(2) , 0.0, 0.0]), decimal=3)
        np.testing.assert_array_almost_equal(lattice.a2, np.array([0, a, 0]), decimal=3)


        
    def test_hex_lattice(self):
        a = 3.84
        a1, a2 = Lattice.hex_lattice(a)
        np.testing.assert_array_almost_equal(a1, np.array([3.84, 0.0, 0.0]))
        np.testing.assert_array_almost_equal(a2, np.array([3.84*0.5, 3.325, 0.0]), decimal=3)

    def test_rotate(self):
        a1 = [1.0, 0.0]
        a2 = [0.0, 1.0]
        lattice = Lattice(a1, a2)
        lattice.rotate(90)
        np.testing.assert_array_almost_equal(lattice.a1, np.array([0.0, 1.0, 0.0]), decimal=4)
        np.testing.assert_array_almost_equal(lattice.a2, np.array([-1.0, 0.0, 0.0]), decimal=4)
        lattice.rotate(-90)
        np.testing.assert_array_almost_equal(lattice.a1, np.array([1.0, 0.0, 0.0]), decimal=4)
        np.testing.assert_array_almost_equal(lattice.a2, np.array([0.0, 1.0, 0.0]), decimal=4)
        
    
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
        lattice_copy = copy.copy(lattice)
        lattice_deepcopy = copy.deepcopy(lattice)
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
        lattice.plot_inverse()

if __name__ == '__main__':
    unittest.main()
