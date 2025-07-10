from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from xrheed.plugins import LoadRheedBase


class LoadPlugin(LoadRheedBase):

    TOLERATED_EXTENSIONS: ClassVar[set[str]] = {".raw"}

    ATTRS = {
        "plugin": "UMCS DSNP ARPES raw",
        "screen_sample_distance": 309.2,  # mm
        "screen_scale": 9.04,  # pixels per mm
        "screen_center_x": 76.0,  # horizontal center of an image
        "screen_center_y": 15.0,  # shadow edge position
        "beam_energy": 18.6 * 1000,  # eV
    }

    def load_single_image(
        self,
        file_path: Path | str,
        plugin_name: str = "",
        **kwargs,
    ):

        if not self.is_file_accepted(file_path):
            print("File not accepted")

        px_to_mm = self.ATTRS["screen_scale"]

        # TODO is should be possible to provide this via kwargs
        image_size = [1038, 1388]

        with Path(file_path).open(mode="r") as file:
            image = (
                np.fromfile(file, dtype=">u2").reshape(*image_size).astype(np.uint16)
            )

        image = (image / 256).astype(np.uint8)

        height, width = image_size
        x_coords = np.arange(width) / px_to_mm
        y_coords = np.arange(height) / px_to_mm

        dims = ["y", "x"]

        x_coords -= self.ATTRS["screen_center_x"]
        y_coords = self.ATTRS["screen_center_y"] - y_coords

        image = image.astype(np.uint8)

        # Flip the y_coords and image vertically to match new y_coords
        y_coords = np.flip(y_coords)
        image = np.flipud(image)

        coords: dict[str, NDArray[np.floating]] = {
            "y": y_coords,
            "x": x_coords,
        }
        attrs = self.ATTRS

        # Create xarray DataArray
        data_array = xr.DataArray(
            image,
            coords=coords,
            dims=dims,
            attrs=attrs,
        )

        return data_array
