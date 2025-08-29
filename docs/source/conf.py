import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))

project = "xRHEED"
author = "Marek Kopciuszynski"
release = "0.1.0"

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

