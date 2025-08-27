import logging

from . import xarray_accessors as xarray_accessors

# Configure logging
logger = logging.getLogger(__name__)
logger.info("xrheed package initialized successfully. Accessors registered.")


# Show a welcome message if running in a Jupyter notebook
def _in_jupyter():
    try:
        from IPython import get_ipython

        shell = get_ipython()
        if shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell":
            return True
    except Exception:
        pass
    return False


if _in_jupyter():
    print(
        "\n🎉 xrheed loaded! \nDocumentation: https://xrheed.readthedocs.io/en/latest/\n"
    )
