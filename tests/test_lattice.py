import unittest
import numpy as np
from xrheed.ewald.lattice import Lattice

class TestLattice(unittest.TestCase):

    def test_initialization(self):
        a1 = [1.0, 0.0]
        a2 = [0.0, 1.0]
        lattice = Lattice(a1, a2)
        np.testing.assert_array_equal(lattice.a1, np.array([1, 0, 0]))
        np.testing.assert_array_equal(lattice.a2, np.array([0, 1, 0]))

    def test_from_bulk_cubic(self):
        a = 5.43
        lattice = Lattice.from_bulk_cubic(a=a, cubic_type="FCC", plane="111")
        np.testing.assert_array_almost_equal(lattice.a1, np.array([3.84, 0.0, 0]), decimal=3)
        np.testing.assert_array_almost_equal(lattice.a2, np.array([3.84*0.5, 3.325, 0]), decimal=3)
        
    def test_hex_lattice(self):
        a = 3.84
        a1, a2 = Lattice.hex_lattice(a)
        np.testing.assert_array_almost_equal(a1, np.array([3.84, 0.0, 0]))
        np.testing.assert_array_almost_equal(a2, np.array([3.84*0.5, 3.325, 0]), decimal=3)

    def test_rotate(self):
        a1 = [1.0, 0.0]
        a2 = [0.0, 1.0]
        lattice = Lattice(a1, a2)
        lattice.rotate(90)
        np.testing.assert_array_almost_equal(lattice.a1, np.array([0.0, 1.0, 0.0]), decimal=4)
        np.testing.assert_array_almost_equal(lattice.a2, np.array([-1.0, 0.0, 0.0]), decimal=4)
    
    def test_invalid_vector(self):
        with self.assertRaises(ValueError):
            Lattice([1], [0, 1])
        with self.assertRaises(ValueError):
            Lattice([1, 0, 0, 0], [0, 1])

if __name__ == '__main__':
    unittest.main()
