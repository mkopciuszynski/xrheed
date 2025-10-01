"""
RHEED Data Loader

Provides a unified API to load RHEED images either via plugins or manually.
"""

import logging
from pathlib import Path
from typing import Optional, Union, Sequence, cast

import numpy as np
import xarray as xr
from numpy.typing import NDArray
from PIL import Image

from .plugins import PLUGINS

CANONICAL_STACK_DIMS = {"alpha", "beta", "coverage", "time", "temperature"}

__all__ = ["load_data"]

logger = logging.getLogger(__name__)


def load_data(
    path: Union[str, Path, Sequence[Union[str, Path]]],
    plugin: Optional[str] = None,
    *,
    screen_sample_distance: Optional[float] = None,
    screen_scale: Optional[float] = None,
    beam_energy: Optional[float] = None,
    screen_center_sx_px: Optional[int] = None,
    screen_center_sy_px: Optional[int] = None,
    alpha: float = 0.0,
    beta: float = 2.0,
    stack_dim: Optional[str] = None,
    stack_coords: Optional[Sequence[float]] = None,
    **kwargs,
) -> xr.DataArray:
    """
    Load a RHEED image (or a stack of images) using either a plugin or manual parameters.

    Parameters
    ----------
    path : str | Path | list[str|Path]
        File path (single image) or list of files (stacked images).
    plugin : str, optional
        Name of plugin to use. If None, manual mode is assumed.
    screen_sample_distance, screen_scale, beam_energy : float
        Required in manual mode.
    screen_center_sx_px, screen_center_sy_px : int, optional
        Optional centers in px (default: image midpoints).
    alpha, beta : float
        Optional angles.
    stack_dim : str, optional
        New dimension name when stacking multiple files.
    stack_coords : array-like, optional
        Coordinates for the new dimension.

    Returns
    -------
    xarray.DataArray
        Image data with coordinates and attributes.
    """

    # --- Multi-file case ---
    if isinstance(path, (list, tuple)):
        print("This is not fully implemented now!")

        if plugin is None:
            raise ValueError("Multi-file loading is only supported with plugins.")
        logger.info(f"Loading {len(path)} files with plugin={plugin}")
        arrays = [load_data(p, plugin=plugin, **kwargs) for p in path]

        if stack_dim is None:
            raise ValueError("stack_dim must be provided when loading multiple files.")

        if stack_dim not in CANONICAL_STACK_DIMS:
            logger.warning(
                f"Non-standard stack dimension '{stack_dim}'. "
                f"Consider using one of {sorted(CANONICAL_STACK_DIMS)} for consistency."
            )

        da = xr.concat(arrays, dim=stack_dim)
        if stack_coords is not None:
            da = da.assign_coords({stack_dim: stack_coords})
        return da

    # --- Single-file case ---
    path = cast(str | Path, path)
    path = Path(path)

    if plugin is not None:
        plugin_cls = PLUGINS[plugin]
        plugin_instance = plugin_cls()
        if not plugin_instance.is_file_accepted(path):
            raise ValueError(
                f"File {path} not accepted by plugin '{plugin}'. "
                f"Allowed extensions: {plugin_cls.TOLERATED_EXTENSIONS}"
            )
        return plugin_instance.load_single_image(path, **kwargs)

    # --- Single-file case - manual mode ---
    else:
    
        assert screen_scale is not None, "screen_scale must be provided in manual mode"
        assert (
            screen_sample_distance is not None
        ), "screen_scale must be provided in manual mode"
        assert beam_energy is not None, "screen_scale must be provided in manual mode"

        # Load image (bmp/png/tiff/…)
        image = Image.open(path).convert("L")
        image_np: NDArray[np.uint8] = np.array(image, dtype=np.uint8)

        h: int
        w: int
        h, w = image_np.shape

        if screen_center_sx_px is None:
            screen_center_sx_px = w // 2
        if screen_center_sy_px is None:
            screen_center_sy_px = h // 2

        sx = (np.arange(w) - screen_center_sx_px) / screen_scale
        sy = (screen_center_sy_px - np.arange(h)) / screen_scale

        sy = np.flip(sy)
        image_np = np.flipud(image_np)

        coords = {"sy": sy, "sx": sx}
        attrs = dict(
            plugin="manual",
            screen_sample_distance=screen_sample_distance,
            screen_scale=screen_scale,
            screen_center_sx_px=screen_center_sx_px,
            screen_center_sy_px=screen_center_sy_px,
            beam_energy=beam_energy,
            alpha=alpha,
            beta=beta,
            file_name=path.name,
        )

        return xr.DataArray(image_np, coords=coords, dims=["sy", "sx"], attrs=attrs)
