import unittest
from pathlib import Path

import xarray as xr
import xrheed

from xrheed.preparation import high_pass_filter


class TestHighPassFilter(unittest.TestCase):
    """Tests for high-pass filtering on RHEED data."""

    def setUp(self):
        self.base = Path(__file__).parent / "data"

    def _load_image(self, fname: str):
        return xrheed.load_data(self.base / fname, plugin="dsnp_arpes_raw")

    def test_filter_single_image(self):
        """Filter should work on a single 2D image."""
        img = self._load_image("Si_111_7x7_112_phi_00.raw")

        filtered = high_pass_filter(img, sigma=1.5, threshold=0.5)

        # shape + dims preserved
        self.assertEqual(filtered.shape, img.shape)
        self.assertEqual(filtered.dims, img.dims)

        # dtype preserved
        self.assertEqual(filtered.dtype, img.dtype)

        # values changed
        self.assertFalse((filtered.values == img.values).all())

        # output non-negative
        self.assertGreaterEqual(filtered.values.min(), 0)

    def test_filter_stack(self):
        """Filter should work on stacked data."""
        img = self._load_image("Si_111_7x7_112_phi_00.raw")

        stack = xr.concat([img, img, img], dim="frame")

        filtered = high_pass_filter(stack, sigma=1.5, threshold=0.5)

        self.assertEqual(filtered.shape, stack.shape)
        self.assertEqual(filtered.dims, stack.dims)
        self.assertEqual(filtered.dtype, stack.dtype)

    def test_coords_preserved(self):
        """Custom coordinates should be preserved."""
        img = self._load_image("Si_111_7x7_112_phi_00.raw")

        img = img.assign_coords(test_coord=("sy", img.sy.values))

        filtered = high_pass_filter(img)

        self.assertIn("test_coord", filtered.coords)

    def test_uint16_support(self):
        """Filter should support uint16 images."""
        img = self._load_image("Si_111_7x7_112_phi_00.raw")

        img16 = img.astype("uint16")

        filtered = high_pass_filter(img16)

        self.assertEqual(filtered.dtype, img16.dtype)

    def test_filter_effect_strength(self):
        """Larger sigma should produce stronger high-pass effect."""
        img = self._load_image("Si_111_7x7_112_phi_00.raw")

        f_small = high_pass_filter(img, sigma=0.5)
        f_large = high_pass_filter(img, sigma=3.0)

        # stronger smoothing → stronger subtraction → higher variance
        self.assertGreater(f_large.var().item(), f_small.var().item())


if __name__ == "__main__":
    unittest.main()
