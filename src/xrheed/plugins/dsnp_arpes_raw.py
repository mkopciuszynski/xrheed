import logging
from pathlib import Path
from typing import ClassVar

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from xrheed.plugins import LoadRheedBase, rheed_plugin

logger = logging.getLogger(__name__)


@rheed_plugin("dsnp_arpes_raw")
class LoadPlugin(LoadRheedBase):
    """Plugin to load UMCS DSNP ARPES raw RHEED images."""

    TOLERATED_EXTENSIONS: ClassVar[set[str]] = {".raw"}

    ATTRS: ClassVar[dict[str, float | str]] = {
        "plugin": "UMCS DSNP ARPES raw",
        "screen_sample_distance": 309.2,  # mm
        "screen_scale": 9.04,  # pixels per mm
        "screen_center_sx_px": 740,  # horizontal center of an image in px
        "screen_center_sy_px": 155,  # vertical center (shadow edge) in px
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
        """
        Load a single RHEED image from a raw binary file.

        Parameters
        ----------
        file_path : Path or str
            Path to the raw RHEED image file.
        plugin_name : str
            Plugin name (unused here, for compatibility with base class).
        **kwargs
            Additional arguments:
            - image_size: optional list [height, width] for non-standard image dimensions.

        Returns
        -------
        xr.DataArray
            The loaded RHEED image as an xarray DataArray with proper coordinates and attributes.
        """
        file_path = Path(file_path)

        if not self.is_file_accepted(file_path):
            raise ValueError(f"File not accepted: {file_path}")

        px_to_mm = float(self.ATTRS["screen_scale"])

        # Allow custom image size via kwargs
        image_size = kwargs.get("image_size", [1038, 1388])
        height, width = image_size

        # Load raw data
        with file_path.open("rb") as file:
            image_16bit: NDArray[np.uint16] = np.fromfile(file, dtype=">u2").reshape(
                *image_size
            )

        # Convert to 8-bit for memory efficiency
        image = (image_16bit / 256).astype(np.uint8)

        # Generate coordinates
        sx_coords: NDArray[np.float64] = np.arange(width, dtype=np.float64)
        sy_coords: NDArray[np.float64] = np.arange(height, dtype=np.float64)

        # Shift coordinates to center
        sx_coords -= float(self.ATTRS["screen_center_sx_px"])
        sy_coords = float(self.ATTRS["screen_center_sy_px"]) - sy_coords

        # Convert from pixels to mm
        sx_coords /= px_to_mm
        sy_coords /= px_to_mm

        # Flip vertically to match new y coordinates
        sy_coords = np.flip(sy_coords)
        image = np.flipud(image)

        coords: dict[str, NDArray[np.floating]] = {
            "sy": sy_coords,
            "sx": sx_coords,
        }
        dims = ["sy", "sx"]
        attrs = self.ATTRS.copy()  # Copy to avoid modifying the class-level dictionary

        # Log the loading
        logger.info(f"Loaded image {file_path} with shape {image.shape}")

        # Create xarray DataArray
        data_array = xr.DataArray(
            data=image,
            coords=coords,
            dims=dims,
            attrs=attrs,
        )

        return data_array
