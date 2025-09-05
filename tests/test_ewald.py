import unittest
from pathlib import Path

from xrheed.io import load_data
from xrheed.kinematics.ewald import Ewald
from xrheed.kinematics.lattice import Lattice


class TestEwald(unittest.TestCase):
    def setUp(self):
        test_data_path = Path(__file__).parent / "data" / "Si_111_7x7_112_phi_00.raw"
        self.rheed_image = load_data(test_data_path, plugin="dsnp_arpes_raw")

    def test_ewald_basic(self):
        # Create a simple lattice
        a1 = [3.84, 0]
        a2 = [1.92, 3.325]
        lattice = Lattice(a1, a2)

        rheed_image = self.rheed_image

        # Instantiate Ewald and run calculation
        ewald = Ewald(lattice, rheed_image)
        ewald.calculate_ewald()

        # Check that spot arrays are created and not empty
        assert hasattr(ewald, "ew_sx")
        assert hasattr(ewald, "ew_sy")
        assert len(ewald.ew_sx) > 0
        assert len(ewald.ew_sy) > 0


if __name__ == "__main__":
    unittest.main()
