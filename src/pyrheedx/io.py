"""Provides the RHEED images loader

TODO: Add direcotry load for many files
"""

from __future__ import annotations

from pathlib import Path

import xarray as xr

from .plugins import load_single_image

__all__ = (
    "load_data",
)

def load_data(
    path: str | Path,
    plugin: str | None = None,
    **kwargs,
) -> xr.DataArray:
    
    if path is None:
        print("You have to prowide the path.")

    print(plugin)
    print(path)

    path = Path(path).absolute()

    if path.is_file():
        return load_single_image(path, plugin)

    elif path.is_dir():
        print(f"{path} is a directory.")
        print("Not implemeneted yet.")

    else:
        print(f"{path} does not exist.")


