"""Provides the RHEED images loader

TODO: Add directory load for many files
"""

from __future__ import annotations

import logging
from pathlib import Path

import xarray as xr

from .plugins import load_single_image

__all__ = ("load_data",)

logger = logging.getLogger(__name__)


def load_data(
    path: str | Path,
    plugin: str | None = None,
    **kwargs,
) -> xr.DataArray:
    """
    Load RHEED data from a file or directory.

    Parameters:
        path (str | Path): Path to the file or directory.
        plugin (str | None): Plugin to use for loading the data.
        **kwargs: Additional arguments passed to the plugin loader.

    Returns:
        xr.DataArray: The loaded RHEED data.

    Raises:
        ValueError: If the path is None or invalid.
        NotImplementedError: If the path is a directory (not yet implemented).
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
