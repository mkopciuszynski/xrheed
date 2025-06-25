from __future__ import annotations


from typing import ClassVar, Type
from pathlib import Path
import xarray as xr
import importlib


class LoadRheedBase:

    TOLERATED_EXTENSIONS: ClassVar[set[str]] = {
        ".png",
        ".raw",
        ".bmp",
    }

    @classmethod
    def is_file_accepted(
        cls,
        file: str | Path,
    ) -> bool:
        """Determines whether this loader can load this file."""
        if Path(file).exists() and Path(file).is_file():
            p = Path(file)

            if p.suffix not in cls.TOLERATED_EXTENSIONS:
                return False
        return True

    def load_single_image(
        self,
        file_path: Path,
        plugin_name: str = "",
        **kwargs,
    ) -> xr.DataArray:
        """Hook for loading a single frame of data.

        This always needs to be overridden in subclasses to handle data appropriately.
        """
        if file_path:
            msg = "You need to define load_single_image."
            raise NotImplementedError(msg)
        return xr.DataArray()


def load_single_image(image_path: Path, plugin_name: str, **kwargs) -> xr.DataArray:
    """Load a single image using the specified plugin."""
    plugin_cls = get_plugin_class(plugin_name)
    plugin_instance = plugin_cls()
    return plugin_instance.load_single_image(image_path, **kwargs)


def load_many_images(
    image_paths: list[Path], plugin_name: str, **kwargs
) -> list[xr.DataArray]:
    pass


def get_plugin_class(plugin_name: str) -> Type[LoadRheedBase]:
    """Dynamically load the plugin class by name."""
    try:
        module = importlib.import_module(f"xrheed.plugins.{plugin_name}")
        if hasattr(module, "LoadPlugin"):
            return module.LoadPlugin
        else:
            raise ImportError(
                f"The plugin '{plugin_name}' does not have a 'Plugin' class."
            )
    except ImportError as e:
        raise ImportError(f"Could not load plugin '{plugin_name}': {e}")
