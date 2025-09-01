Introduction
============
What is RHEED
-------------

**Reflection High Energy Electron Diffraction (RHEED)** is an experimental technique for monitoring and controlling crystal surface quality.
A high-energy electron beam (~20 keV) strikes the surface at a grazing angle (< 5°), making the method highly **surface-sensitive**, probing only a few atomic layers.

Project Goals
-------------

**xrheed** provides a flexible and extensible **Python toolkit** for RHEED image analysis:

- Load and preprocess RHEED images
- Generate and analyze intensity profiles
- Overlay predicted diffraction spot positions (kinematic theory & Ewald construction)

.. note::

   **Note:** xrheed is **not a GUI application**. It is designed as an **xarray accessory library** to facilitate analysis in **interactive environments** such as Jupyter notebooks.
