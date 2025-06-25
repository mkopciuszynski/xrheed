import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

project = 'xRHEED'
author = 'mkopciuszynski'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',    # for Google/NumPy docstrings
    'sphinx.ext.mathjax',     # for LaTeX
    'nbsphinx',               # or use 'myst_nb'
]

templates_path = ['_templates']
exclude_patterns = ['_build']

html_theme = 'sphinx_rtd_theme'

# Allow data files to be found by notebooks
nbsphinx_execute = 'always'
nbsphinx_allow_errors = True