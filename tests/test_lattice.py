import unittest
import numpy as np
from pyrheed.ewald.lattice import Lattice

class TestLattice(unittest.TestCase):

    def test_initialization(self):
        a1 = [1.0, 0.0]
        a2 = [0.0, 1.0]
        lattice = Lattice(a1, a2)
        np.testing.assert_array_equal(lattice.a1, np.array([1, 0, 0]))
        np.testing.assert_array_equal(lattice.a2, np.array([0, 1, 0]))

    def test_from_bulk(self):
        lattice = Lattice.from_bulk(5.43, plane='111')
        np.testing.assert_array_almost_equal(lattice.a1, np.array([3.84, 0.0, 0]), decimal=2)
        np.testing.assert_array_almost_equal(lattice.a2, np.array([3.84*0.5, 3.325, 0]), decimal=2)
        

    def test_from_surface(self):
        lattice = Lattice.from_surface(1.0, plane='111')
        np.testing.assert_array_almost_equal(lattice.a1, np.array([1.0, 0.0, 0.0]), decimal=4)
        np.testing.assert_array_almost_equal(lattice.a2, np.array([0.5, 0.8660, 0.0]), decimal=4)

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
