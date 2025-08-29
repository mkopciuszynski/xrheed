# -- Path setup --------------------------------------------------------------
import os
import sys
from pathlib import Path
from datetime import datetime

HERE = Path(__file__).resolve().parent
# Adjust ("..") to ("..", "..") if your docs are under docs/source/
SRC = (HERE / ".." / "src").resolve()
if SRC.is_dir():
    sys.path.insert(0, str(SRC))

# -- Project information -----------------------------------------------------
project = "xRHEED"
author = "Marek Kopciuszynski"
release = "0.1.0"
copyright = f"{datetime.now().year}, {author}"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",   # for API stubs/tables
    "sphinx.ext.napoleon",      # Google/NumPy docstrings
    "sphinx.ext.mathjax",       # LaTeX
    "sphinx.ext.viewcode",      # View source links (optional)
    "myst_parser",              # Markdown (MyST)
    "myst_nb",                  # Jupyter notebooks
    # "sphinx_autodoc_typehints",# optional: nicer type hints
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Parse both RST and MD files
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# MyST options (optional but handy)
myst_enable_extensions = [
    "colon_fence", "deflist", "linkify", "substitution", "tasklist",
]

# Notebooks: render but don't execute (fast)
nb_execution_mode = "auto"  # "off" | "auto" | "force"
nb_render_plugin = "default"

# Autodoc/autosummary settings
autosummary_generate = True
add_module_names = False
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_title = project
