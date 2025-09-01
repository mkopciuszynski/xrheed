import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))

import xrheed

project = "xRHEED"

release = xrheed.__version__
version = ".".join(release.split(".")[:2])

author = "Marek Kopciuszynski"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "myst_nb"
]

templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "sphinx_rtd_theme"

nb_execution_mode = "auto"  # or "off", "cache", "force"

autosummary_generate = True

autodoc_mock_imports = ["xarray", "cupy", "array_api_strict"]

