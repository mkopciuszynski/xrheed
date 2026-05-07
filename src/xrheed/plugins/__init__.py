"""
Plugin system for RHEED data loading.

Design principles:
- Plugins describe how to load: pixels, optional metadata.
- A single canonical constructor creates valid RHEED DataArrays.
- File provenance is always attached automatically.
"""

import abc
import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Set, Type

import numpy as np
import xarray as xr

PLUGINS: Dict[str, Type["LoadRheedBase"]] = {}


def register_plugin(name: str):
    """Decorator to register a new plugin."""

    def decorator(cls):
        PLUGINS[name] = cls
        return cls

    return decorator


class LoadRheedBase(abc.ABC):
    """
    Base class for RHEED plugins.
    """

    TOLERATED_EXTENSIONS: Set[str] = set()
    ATTRS: Dict[str, Any] = {}

    def is_file_accepted(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self.TOLERATED_EXTENSIONS

    @abc.abstractmethod
    def load_single_image(self, file_path: Path, **kwargs) -> xr.DataArray:
        """
        Load a single image and return a canonical RHEED DataArray.

        Implementations MUST call `self.dataarray_from_image(...)`
        exactly once and return its result.
        """
        raise NotImplementedError


    def dataarray_from_image(
        self,
        image_np: np.ndarray,
        *,
        file_path: Optional[Path] = None,
        attrs_override: Optional[Dict[str, Any]] = None,
        flip: bool = True,
    ) -> xr.DataArray:
        """
        Construct a canonical RHEED DataArray from an image.

        Responsibilities:
        - merge default attrs with overrides
        - resolve geometry (including screen center)
        - construct sx / sy coordinates
        - attach file provenance automatically
        """

        if image_np.ndim != 2:
            raise ValueError("RHEED image must be a 2D array")

        # --------------------------------------------------------------
        # 1. Merge attributes
        # --------------------------------------------------------------
        attrs: Dict[str, Any] = dict(self.ATTRS)
        if attrs_override:
            attrs.update(attrs_override)

        # --------------------------------------------------------------
        # 2. Validate required geometry
        # --------------------------------------------------------------
        screen_scale = attrs.get("screen_scale")
        if screen_scale is None:
            raise ValueError(
                "screen_scale must be defined to construct RHEED coordinates"
            )

        px_to_mm = float(screen_scale)
        h, w = image_np.shape

        # --------------------------------------------------------------
        # 3. Resolve screen center (None → image center)
        # --------------------------------------------------------------
        cx = attrs.get("screen_center_sx_px")
        cy = attrs.get("screen_center_sy_px")

        if cx is None:
            cx = w // 2
        if cy is None:
            cy = h // 2

        # --------------------------------------------------------------
        # 4. Construct coordinates
        # --------------------------------------------------------------
        sx = (np.arange(w) - cx) / px_to_mm
        sy = (cy - np.arange(h)) / px_to_mm

        if flip:
            sy = np.flip(sy)
            image_np = np.flipud(image_np)

        # --------------------------------------------------------------
        # 5. Create DataArray
        # --------------------------------------------------------------
        da = xr.DataArray(
            image_np,
            dims=("sy", "sx"),
            coords={"sy": sy, "sx": sx},
            attrs=attrs,
        )

        # --------------------------------------------------------------
        # 6. Attach file metadata (provenance)
        # --------------------------------------------------------------
        if file_path is not None:
            da = self._attach_file_metadata(da, file_path)

        return da

    def _attach_file_metadata(self, da: xr.DataArray, file_path: Path) -> xr.DataArray:
        """Attach file provenance metadata to attrs."""
        da.attrs["file_name"] = file_path.name
        try:
            stat = file_path.stat()
            da.attrs["file_ctime"] = datetime.datetime.fromtimestamp(
                stat.st_mtime
            ).strftime("%Y-%m-%d, %H:%M:%S")
        except Exception:
            pass
        return da