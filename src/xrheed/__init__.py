"""
xRHEED: An xarray-based toolkit for RHEED image analysis.

This package provides tools to:
- Load and preprocess RHEED images
- Extract and analyze intensity profiles
- Transform images to kx-ky space
- Predict and visualize diffraction spot positions using kinematic theory and Ewald construction
xRHEED is designed as an **xarray accessory library** for interactive analysis
in environments such as Jupyter notebooks. It is **not a GUI application**.
"""

import importlib
import logging
import pkgutil
from importlib.metadata import PackageNotFoundError, version

# Import xarray accessors
from . import xarray_accessors as xarray_accessors

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
