"""
RHEED Data Loaders

This module provides functions to load RHEED images from files using registered plugins.
"""

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import xarray as xr
from numpy.typing import NDArray
from PIL import Image

from .plugins import load_single_image

__all__ = ["load_data", "load_data_manual"]

logger = logging.getLogger(__name__)


def load_data(
    path: Union[str, Path],
    plugin: str,
    **kwargs,
) -> xr.DataArray:
    """
    Load a single RHEED image using the specified plugin.

    Parameters
    ----------
    path : str or Path
        Path to the file containing RHEED data.
    plugin : str
        Name of the plugin to use for loading.
    **kwargs : dict
        Additional arguments passed to the plugin loader.

    Returns
    -------
    xr.DataArray
        The loaded RHEED image.

    Raises
    ------
    ValueError
        If the path is invalid or does not exist.
    NotImplementedError
        If the path is a directory (directory loading not yet implemented).
    """
    if not path:
        raise ValueError("You must provide a valid path.")

    path = Path(path).absolute()
    logger.info(f"Loading data from: {path}")
    logger.debug(f"Using plugin: {plugin}")

    if path.is_file():
        logger.info(f"Detected file: {path}")
        return load_single_image(path, plugin, **kwargs)

    elif path.is_dir():
        logger.warning(f"Directory loading is not implemented yet: {path}")
        raise NotImplementedError(
            "Loading data from directories is not implemented yet."
        )

    else:
        logger.error(f"Path does not exist: {path}")
        raise ValueError(f"The specified path does not exist: {path}")


def load_data_manual(
    path: Union[str, Path],
    *,
    screen_sample_distance: float,
    screen_scale: float,
    beam_energy: float,
    screen_center_sx_px: Optional[int] = None,
    screen_center_sy_px: Optional[int] = None,
    alpha: float = 0.0,
    beta: float = 2.0,
) -> xr.DataArray:
    """
    Manually load a RHEED image without using a plugin.

    This is the fallback loader for cases where no plugin is available.
    The user must provide the essential parameters.

    Parameters
    ----------
    path : str | Path
        Path to the image file (BMP, PNG, TIFF, etc.).
    screen_sample_distance : float
        Distance from sample to screen [mm].
    screen_scale : float
        Scaling factor [pixels per mm].
    beam_energy : float
        Beam energy [eV].
    screen_center_sx_px : int, optional
        Horizontal center of the image in pixels.
        Defaults to image midpoint if not provided.
    screen_center_sy_px : int, optional
        Vertical center of the image in pixels.
        Defaults to image midpoint if not provided.
    alpha : float, optional
        Azimuthal angle, by default 0.0.
    beta : float, optional
        Incident angle, by default 2.0.

    Returns
    -------
    xarray.DataArray
        Image data with physical coordinates and attributes.

    Raises
    ------
    ValueError
        If any of the required calibration parameters are missing.
    """
    path = Path(path)

    # Ensure required parameters are present
    for key, val in {
        "screen_sample_distance": screen_sample_distance,
        "screen_scale": screen_scale,
        "beam_energy": beam_energy,
    }.items():
        if val is None:
            raise ValueError(f"Missing required parameter: '{key}'")

    # Load image using Pillow (convert to grayscale, ensure np.uint8)
    try:
        image = Image.open(path).convert("L")
    except Exception as e:
        raise ValueError(f"Cannot load image file '{path}': {e}")

    image_np: NDArray[np.uint8] = np.array(image, dtype=np.uint8)

    height: int
    width: int
    height, width = image_np.shape

    # Default centers if not given
    if screen_center_sx_px is None:
        screen_center_sx_px = width // 2
    if screen_center_sy_px is None:
        screen_center_sy_px = height // 2

    # Coordinates in mm
    sx_coords: NDArray[np.float64] = (
        np.arange(width, dtype=np.float64) - screen_center_sx_px
    ) / screen_scale
    sy_coords: NDArray[np.float64] = (
        screen_center_sy_px - np.arange(height, dtype=np.float64)
    ) / screen_scale

    # Flip vertically to match coordinate orientation
    sy_coords = np.flip(sy_coords)
    image_np = np.flipud(image_np)

    coords: dict[str, NDArray[np.floating]] = {
        "sy": sy_coords,
        "sx": sx_coords,
    }
    dims = ["sy", "sx"]

    attrs: dict[str, float | str] = {
        "plugin": "manual",
        "screen_sample_distance": screen_sample_distance,
        "screen_scale": screen_scale,
        "screen_center_sx_px": screen_center_sx_px,
        "screen_center_sy_px": screen_center_sy_px,
        "beam_energy": beam_energy,
        "alpha": alpha,
        "beta": beta,
    }

    return xr.DataArray(
        data=image_np,
        coords=coords,
        dims=dims,
        attrs=attrs,
    )
