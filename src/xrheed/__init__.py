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

import logging
from importlib.metadata import PackageNotFoundError, version

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
        if shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell":
            return True
    except Exception:
        pass
    return False


# Show a welcome message in Jupyter
if _in_jupyter():
    print(
        f"\n🎉 xrheed v{__version__} loaded!"
        "\n📖 Documentation: https://xrheed.readthedocs.io/en/latest/\n"
    )
