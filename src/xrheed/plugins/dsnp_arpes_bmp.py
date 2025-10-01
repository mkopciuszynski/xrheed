from pathlib import Path

import numpy as np
import xarray as xr
from numpy.typing import NDArray
from PIL import Image

from . import LoadRheedBase, register_plugin


@register_plugin("dsnp_arpes_bmp")
class DsnpArpesBmpPlugin(LoadRheedBase):
    """Plugin to load UMCS DSNP ARPES BMP RHEED images."""

    TOLERATED_EXTENSIONS = {".bmp"}

    ATTRS = {
        "plugin": "UMCS DSNP ARPES bmp",
        "screen_sample_distance": 309.2,  # mm
        "screen_scale": 9.04,  # pixels per mm
        "screen_center_sx_px": 749,  # horizontal center of an image in px
        "screen_center_sy_px": 70,  # vertical center (shadow edge) in px
        "beam_energy": 18_600,  # eV
        "alpha": 0.0,  # azimuthal angle
        "beta": 2.0,  # incident angle
    }

    def load_single_image(self, file_path: Path, **kwargs) -> xr.DataArray:
        # Load BMP image using Pillow and convert to numpy array
        image = Image.open(file_path).convert("L")  # Convert to grayscale
        image_np: NDArray[np.uint8] = np.array(image)

        da = self.dataarray_from_image(image_np)

        return self.add_file_metadata(da, file_path)
