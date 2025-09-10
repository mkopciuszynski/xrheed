import logging
from pathlib import Path
from typing import ClassVar

import numpy as np
import xarray as xr
from numpy.typing import NDArray
from PIL import Image

from xrheed.plugins import LoadRheedBase, rheed_plugin

logger = logging.getLogger(__name__)


@rheed_plugin("dsnp_arpes_bmp")
class LoadPlugin(LoadRheedBase):
    """Plugin to load UMCS DSNP ARPES BMP RHEED images."""

    TOLERATED_EXTENSIONS: ClassVar[set[str]] = {".bmp"}

    ATTRS: ClassVar[dict[str, float | str]] = {
        "plugin": "UMCS DSNP ARPES bmp",
        "screen_sample_distance": 309.2,  # mm
        "screen_scale": 9.04,  # pixels per mm
        "screen_center_sx_px": 749,  # horizontal center of an image in px
        "screen_center_sy_px": 70,  # vertical center (shadow edge) in px
        "beam_energy": 18.6 * 1000,  # eV
        "alpha": 0.0,  # azimuthal angle
        "beta": 2.0,  # incident angle
    }

    def load_single_image(
        self,
        file_path: Path | str,
        plugin_name: str = "",
        **kwargs,
    ) -> xr.DataArray:
        file_path = Path(file_path)

        if not self.is_file_accepted(file_path):
            raise ValueError(f"File not accepted: {file_path}")

        px_to_mm = float(self.ATTRS["screen_scale"])

        # Load BMP image using Pillow
        image = Image.open(file_path).convert("L")  # Convert to grayscale
        image_np: NDArray[np.uint8] = np.array(image)

        height, width = image_np.shape

        # Generate coordinates
        sx_coords = np.arange(width, dtype=np.float64)
        sy_coords = np.arange(height, dtype=np.float64)

        # Shift coordinates to center
        sx_coords -= float(self.ATTRS["screen_center_sx_px"])
        sy_coords = float(self.ATTRS["screen_center_sy_px"]) - sy_coords
        # Convert from pixels to mm
        sx_coords /= px_to_mm
        sy_coords /= px_to_mm

        # Flip vertically to match new y coordinates
        sy_coords = np.flip(sy_coords)
        image_np = np.flipud(image_np)

        coords = {"sy": sy_coords, "sx": sx_coords}
        dims = ["sy", "sx"]
        attrs = self.ATTRS.copy()

        logger.info(f"Loaded BMP image {file_path} with shape {image_np.shape}")

        return xr.DataArray(
            data=image_np,
            coords=coords,
            dims=dims,
            attrs=attrs,
        )
