import os
import sys
import tomllib
from datetime import datetime


sys.path.insert(0, os.path.abspath("../../."))

project = "xrheed"

# Read version from pyproject.toml
with open(os.path.join(os.path.dirname(__file__), "../../pyproject.toml"), "rb") as f:
    pyproject = tomllib.load(f)

release = pyproject["project"]["version"]
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
