"""
Plugin system for RHEED data loading.
"""

import abc
import datetime
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Set, Type

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

PLUGINS: Dict[str, Type["LoadRheedBase"]] = {}


def register_plugin(name: str):
    """Decorator to register a new plugin."""

    def decorator(cls):
        PLUGINS[name] = cls
        logger.info(f"Registered RHEED plugin: {name}")
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
    def load_single_image(self, file_path: Path, **kwargs) -> xr.DataArray: ...

    def add_file_metadata(self, da: xr.DataArray, file_path: Path) -> xr.DataArray:
        """Attach common metadata (file name, creation time) to attrs."""
        da.attrs["file_name"] = file_path.name
        try:
            stat = Path(file_path).stat()
            # Use st_birthtime if available, otherwise fallback to st_ctime
            ctime = getattr(stat, "st_birthtime", stat.st_ctime)
            da.attrs["file_ctime"] = datetime.datetime.fromtimestamp(ctime).strftime(
                "%Y-%m-%d, %H:%M:%S"
            )
        except Exception:
            pass
        return da

    def dataarray_from_image(
        self,
        image_np: np.ndarray,
        attrs_override: Optional[Dict[str, Any]] = None,
        flip: bool = True,
    ) -> xr.DataArray:
        """
        Helper to create DataArray from a numpy image, using ATTRS scaling and centers.
        Plugins may use or override this.
        """
        px_to_mm = float(self.ATTRS["screen_scale"])
        h, w = image_np.shape

        sx = (np.arange(w) - self.ATTRS.get("screen_center_sx_px", w // 2)) / px_to_mm
        sy = (self.ATTRS.get("screen_center_sy_px", h // 2) - np.arange(h)) / px_to_mm

        if flip:
            sy = np.flip(sy)
            image_np = np.flipud(image_np)

        coords = {"sy": sy, "sx": sx}
        attrs = {**self.ATTRS, **(attrs_override or {})}

        return xr.DataArray(image_np, coords=coords, dims=["sy", "sx"], attrs=attrs)
