import logging
from importlib.metadata import version, PackageNotFoundError

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
