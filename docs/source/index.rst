xRHEED
======

.. image:: https://github.com/mkopciuszynski/xrheed/actions/workflows/ci.yml/badge.svg?branch=main
   :target: https://github.com/mkopciuszynski/xrheed/actions/workflows/ci.yml
   :alt: CI

.. image:: https://readthedocs.org/projects/xrheed/badge/
   :target: https://xrheed.readthedocs.io/
   :alt: Documentation Status

.. image:: https://img.shields.io/pypi/v/xrheed.svg
   :target: https://pypi.org/project/xrheed/
   :alt: PyPI version

.. image:: https://img.shields.io/badge/packaging-uv-blue
   :target: https://github.com/astral-sh/uv
   :alt: Package manager: uv

.. image:: https://img.shields.io/badge/linter-ruff-46a2f1.svg?logo=ruff
   :target: https://github.com/astral-sh/ruff
   :alt: Linter: ruff

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

.. image:: https://zenodo.org/badge/963155496.svg
   :target: https://doi.org/10.5281/zenodo.17099751
   :alt: DOI

Welcome to the **xRHEED** documentation!

**xRHEED** is an **xarray-based toolkit** for analyzing 
**Reflection High Energy Electron Diffraction (RHEED)** images.

Explore the :doc:`introduction` to learn about the method and project goals.  
Check out the :doc:`usage` guide for a fast start.  
See the :ref:`example-notebooks` for interactive tutorials.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   introduction
   installation
   usage
   geometry
   kinematic
   loaders

.. _example-notebooks:
.. toctree::
   :maxdepth: 2
   :caption: Example Notebooks
   
   notebooks/getting_started
   notebooks/diffraction_profiles
   notebooks/ewald_construction
   notebooks/ewald_kxky_transformation
   notebooks/ewald_kxky_transformation_advanced
   notebooks/ewald_spot_matching

.. toctree::
   :maxdepth: 1
   :titlesonly:
   :caption: API Reference

   xrheed


.. toctree::
   :maxdepth: 2
   :caption: Development

   development/changelog
   development/contributing
