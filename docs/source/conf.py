import os
import sys
import types
import warnings
from datetime import datetime

from tqdm import TqdmWarning

sys.path.insert(0, os.path.abspath("../../."))

# Disable tqdm bars during docs build
os.environ["TQDM_DISABLE"] = "1"  # disables all tqdm bars
# Prevent notebook widget imports from causing warnings
sys.modules["ipywidgets"] = types.ModuleType("ipywidgets")
sys.modules["IPython.html.widgets"] = types.ModuleType("IPython.html.widgets")
# Suppress tqdm warning about missing IProgress
warnings.filterwarnings("ignore", category=TqdmWarning)

project = "xRHEED"

# Import the package to get version
try:
    import xrheed

    release = xrheed.__version__
except ImportError:
    release = "0.0.0"

# Short X.Y version
version = ".".join(release.split(".")[:2])

author = "Marek Kopciuszynski"
year = datetime.now().year
copyright = f"{year}, {author}"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "myst_nb",
]

myst_enable_extensions = ["dollarmath"]

templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "sphinx_rtd_theme"

nb_execution_mode = "auto"  # or "off", "cache", "force"

autosummary_generate = True

# Show full docstrings
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "inherited-members": True,
    "special-members": "__init__",
}

# Optional: collapse module path in references
add_module_names = False

autodoc_typehints = "none"
autodoc_mock_imports = ["cupy", "array_api_strict"]
