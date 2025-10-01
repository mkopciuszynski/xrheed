"""
xRHEED: An xarray-based toolkit for RHEED image analysis.
"""

import importlib
import logging
import pkgutil
from importlib.metadata import PackageNotFoundError, version

# Expose top-level API
from . import xarray_accessors  # noqa: F401 (registers accessors)
from .loaders import load_data

__all__ = ["load_data", "__version__"]

# Package version (from setuptools_scm if installed, otherwise fallback)
try:
    __version__ = version("xrheed")
except PackageNotFoundError:
    __version__ = "0.0.0"

# Configure logging
logger = logging.getLogger(__name__)

# Auto-discover plugins
def _discover_plugins():
    try:
        import xrheed.plugins

        for _, module_name, is_pkg in pkgutil.iter_modules(xrheed.plugins.__path__):
            if not is_pkg:
                importlib.import_module(f"xrheed.plugins.{module_name}")
    except Exception as e:
        logger.warning(f"Plugin discovery failed: {e}")

_discover_plugins()

# Optional: friendly message in notebooks
def _in_jupyter() -> bool:
    try:
        from IPython import get_ipython
        shell = get_ipython()
        return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False

if _in_jupyter():
    print(f"\n🎉 xrheed v{__version__} loaded!")
