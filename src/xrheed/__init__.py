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

# Configure logging
logger = logging.getLogger("xrheed")

# Package version (from setuptools_scm if installed, otherwise fallback)
try:
    __version__ = version("xrheed")
    logger.info("xrheed version detected: %s", __version__)
except PackageNotFoundError:
    __version__ = "0.0.0"
    logger.warning(
        "xrheed version could not be determined; using fallback %s.", __version__
    )


# Auto-discover plugins
def _discover_plugins():
    logger.info("Starting plugin discovery for xrheed...")
    try:
        import xrheed.plugins

        found = False
        for _, module_name, is_pkg in pkgutil.iter_modules(xrheed.plugins.__path__):
            if not is_pkg:
                importlib.import_module(f"xrheed.plugins.{module_name}")
                logger.debug("Loaded plugin module: xrheed.plugins.%s", module_name)
                found = True
        if not found:
            logger.warning("No plugin modules found in xrheed.plugins.")
        else:
            logger.info("Plugin discovery completed successfully.")
    except Exception as e:
        logger.error("Plugin discovery failed: %s", e)


_discover_plugins()


# Optional: friendly message in notebooks
def _in_jupyter() -> bool:
    try:
        from IPython.core.getipython import get_ipython

        shell = get_ipython()
        return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


if _in_jupyter():
    print("\n🎉 xrheed v%s loaded!" % __version__)
