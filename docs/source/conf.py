import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))

project = "xRHEED"
author = "Marek Kopciuszynski"
release = "0.1.0"

tensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # for Google/NumPy docstrings
    "sphinx.ext.mathjax",   # for LaTeX
    "myst_nb",              # handles notebooks and Markdown
]

templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "sphinx_rtd_theme"

# Optional: allow notebooks to execute and show outputs
nbsphinx_execute = "always"
nbsphinx_allow_errors = True