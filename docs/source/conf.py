import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

project = 'xRHEED'
author = 'mkopciuszynski'
release = '0.1.0'

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
]

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'sphinx_rtd_theme'

# MyST-NB options
nb_execution_mode = "off"