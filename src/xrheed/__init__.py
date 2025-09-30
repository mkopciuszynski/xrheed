"""
xRHEED: An xarray-based toolkit for RHEED image analysis.
"""

import importlib
import logging
import pkgutil
from importlib.metadata import PackageNotFoundError, version

# Import xarray accessors
from . import xarray_accessors  # noqa: F401
from .loaders import load_data, load_data_manual

__all__ = ["load_data", "load_data_manual", "__version__"]

# Package version
try:
    __version__ = version("xrheed")
except PackageNotFoundError:
    __version__ = "0.0.0"

# Configure logging
logger = logging.getLogger(__name__)
logger.info(f"xrheed {__version__} initialized successfully. Accessors registered.")


# Check if running inside a Jupyter notebook
def _in_jupyter():
    try:
        from IPython import get_ipython

        shell = get_ipython()
        return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


# Show a welcome message in Jupyter
if _in_jupyter():
    print(f"\n🎉 xrheed v{__version__} loaded!")


# Plugin discovery
def discover_plugins():
    try:
        import xrheed.plugins

        for _, module_name, is_pkg in pkgutil.iter_modules(xrheed.plugins.__path__):
            if not is_pkg:
                importlib.import_module(f"xrheed.plugins.{module_name}")
    except Exception as e:
        logger.warning(f"Plugin discovery failed: {e}")


# Run plugin discovery after all imports
discover_plugins()
