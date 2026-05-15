from pathlib import Path

import numpy as np
import xarray as xr
from PIL import Image

from . import LoadRheedBase, register_plugin


@register_plugin("example_rheed_bmp")
class ExampleRheedBmpPlugin(LoadRheedBase):
    """
    Plugin for loading grayscale BMP RHEED images.

    Contract:
    - ATTRS define defaults only
    - Acquisition parameters (alpha, beta) are attached ONLY if known
    - The returned DataArray is always fully initialized
    """

    TOLERATED_EXTENSIONS = {".bmp"}

    ATTRS = {
        # Instrument / geometry
        "screen_sample_distance": 309.2,  # mm (system constant)
        "screen_scale": 10.0,  # px / mm (required)
        "screen_center_sx_px": None,  # default auto-center if None
        "screen_center_sy_px": None,
        "beam_energy": 19_400,  # eV (if system constant or None)
        "alpha": None,  # azimuthal angle [deg]
        "beta": None,  # incident angle [deg]
    }

    def load_single_image(
        self,
        file_path: Path,
        **kwargs,
    ) -> xr.DataArray:
        # ----------------------------------------------------------
        # 1. Load pixel data logic
        # ----------------------------------------------------------
        image = Image.open(file_path).convert("L")
        image_np = np.asarray(image, dtype=np.uint8)

        # ----------------------------------------------------------
        # 2. Optional metadata extraction
        # ----------------------------------------------------------
        attrs_override = {}

        metadata = self._read_metadata(file_path)

        for key in (
            "beam_energy",
            "alpha",
            "beta",
        ):
            if key in metadata:
                attrs_override[key] = metadata[key]

        # ----------------------------------------------------------
        # 3. Construct canonical DataArray
        # ----------------------------------------------------------
        return self.dataarray_from_image(
            image_np,
            file_path=file_path,
            attrs_override=attrs_override,
        )

    def _read_metadata(self, file_path: Path) -> dict:
        """
        Optional metadata reader.

        Can later be extended to read:
        - sidecar files
        - experimental logs
        - embedded image metadata
        """
        return {}
