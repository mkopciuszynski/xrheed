from pathlib import Path

import numpy as np
import xarray as xr

from . import LoadRheedBase, register_plugin


@register_plugin("dsnp_arpes_raw")
class DsnpArpesRawPlugin(LoadRheedBase):
    """Plugin to load UMCS DSNP ARPES RAW RHEED images."""

    TOLERATED_EXTENSIONS = {".raw"}

    ATTRS = {
        "plugin": "UMCS DSNP ARPES raw",
        "screen_sample_distance": 309.2,  # mm
        "screen_scale": 9.04,  # pixels per mm
        "screen_center_sx_px": 740,
        "screen_center_sy_px": 155,
        "beam_energy": 18_6000,  # eV
        "alpha": 0.0,
        "beta": 2.0,
    }

    def load_single_image(self, file_path: Path, **kwargs) -> xr.DataArray:
        # Load image data into numpy
        raw = np.fromfile(file_path, dtype=">u2").reshape(1038, 1388)
        image_np = (raw / 256).astype(np.uint8)

        da = self.dataarray_from_image(image_np)
        return self.add_file_metadata(da, file_path)
