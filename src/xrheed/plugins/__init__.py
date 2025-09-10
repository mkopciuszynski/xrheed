from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Dict, List, Type

import xarray as xr

__all__ = ["LoadRheedBase", "load_single_image", "load_many_images", "rheed_plugin"]


# -----------------------------
# Abstract Base Class
# -----------------------------
class LoadRheedBase(ABC):
    """Abstract base class for RHEED data loader plugins."""

    # File extensions this plugin can handle
    TOLERATED_EXTENSIONS: ClassVar[set[str]] = set()

    # Central registry: plugin name -> plugin class
    PLUGINS: ClassVar[Dict[str, Type["LoadRheedBase"]]] = {}

    # Attribute keys required in every plugin
    REQUIRED_ATTR_KEYS: ClassVar[set[str]] = {
        "plugin",
        "screen_sample_distance",
        "screen_scale",
        "beam_energy",
    }

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Validate ATTRS
        attrs = getattr(cls, "ATTRS", None)
        if not isinstance(attrs, dict):
            raise TypeError(f"{cls.__name__} must define ATTRS as a dictionary.")

        missing = cls.REQUIRED_ATTR_KEYS - attrs.keys()
        if missing:
            raise ValueError(
                f"{cls.__name__} is missing required ATTRS keys: {missing}"
            )

    @classmethod
    def register_plugin(cls, name: str, plugin_cls: Type["LoadRheedBase"]):
        """Register a plugin class under a given name."""
        if not issubclass(plugin_cls, LoadRheedBase):
            raise TypeError(f"{plugin_cls} must inherit from LoadRheedBase")
        cls.PLUGINS[name] = plugin_cls

    @classmethod
    def get_plugin(cls, name: str) -> Type["LoadRheedBase"]:
        """Retrieve a plugin class by name."""
        if name not in cls.PLUGINS:
            raise ValueError(f"Plugin '{name}' is not registered")
        return cls.PLUGINS[name]

    @classmethod
    def find_plugin_by_extension(cls, ext: str) -> List[Type["LoadRheedBase"]]:
        """Return all plugin classes that can handle a given file extension."""
        ext = ext.lower()
        return [p for p in cls.PLUGINS.values() if ext in p.TOLERATED_EXTENSIONS]

    @classmethod
    def is_file_accepted(cls, file: str | Path) -> bool:
        """Determine whether this loader can handle the file."""
        p = Path(file)
        if not p.exists() or not p.is_file():
            return False
        return p.suffix.lower() in (ext.lower() for ext in cls.TOLERATED_EXTENSIONS)

    @abstractmethod
    def load_single_image(self, file_path: Path, **kwargs) -> xr.DataArray:
        """Load a single RHEED image. Must be implemented by subclasses."""
        pass


# -----------------------------
# Decorator for Plugin Registration
# -----------------------------
def rheed_plugin(name: str):
    """
    Decorator to automatically register a RHEED plugin class.

    Usage:
        @rheed_plugin("plugin_name")
        class LoadPlugin(LoadRheedBase):
            ...
    """

    def wrapper(cls):
        LoadRheedBase.register_plugin(name, cls)
        return cls

    return wrapper


# -----------------------------
# Utility Functions
# -----------------------------
def load_single_image(image_path: Path, plugin_name: str, **kwargs) -> xr.DataArray:
    """Load a single image using the specified plugin."""
    plugin_cls = LoadRheedBase.get_plugin(plugin_name)
    plugin_instance = plugin_cls()
    return plugin_instance.load_single_image(image_path, **kwargs)


def load_many_images(
    image_paths: list[Path], plugin_name: str, **kwargs
) -> list[xr.DataArray]:
    """Load multiple images using the specified plugin."""
    plugin_cls = LoadRheedBase.get_plugin(plugin_name)
    plugin_instance = plugin_cls()
    return [plugin_instance.load_single_image(p, **kwargs) for p in image_paths]
