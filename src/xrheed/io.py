"""
RHEED Data Loader

This module provides functions to load RHEED images from files using registered plugins.

Functions
---------
- load_data(path, plugin, **kwargs): Load a RHEED image from a file.

TODO
----
- Implement loading from directories or multiple files.
"""

import logging
from pathlib import Path
from typing import Union

import xarray as xr

from .plugins import load_single_image

__all__ = ["load_data"]

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
