Introduction
============

🔬 What is RHEED?
-----------------

**Reflection High Energy Electron Diffraction (RHEED)** is an experimental technique widely used in 
surface science and thin film growth.  

In RHEED, a beam of high-energy electrons (typically ~20-30 keV) strikes the sample surface at a grazing 
incidence angle (< 5°). Due to this geometry:

- The electrons interact only with the **topmost atomic layers**, making RHEED extremely 
  **surface-sensitive**.
- The resulting diffraction pattern provides **real-time feedback** about the crystal structure, 
  surface quality, and growth dynamics of thin films.

RHEED is especially useful during **Molecular Beam Epitaxy (MBE)**, where it allows monitoring 
of surface reconstructions, layer-by-layer growth, and lattice parameters.


🎯 Project Goals
----------------

**xRHEED** is a flexible and extensible **Python toolkit** for analyzing RHEED data.  
It is designed as an **xarray accessory library**, meaning it integrates naturally with the 
scientific Python ecosystem.

Main features include:

- 🖼️ **Load and preprocess** RHEED images
- 📈 **Generate and analyze intensity profiles**
- ✨ **Overlay predicted diffraction spot positions** 
  (kinematic theory & Ewald construction)
- 🔄 **Transform images into reciprocal (kx-ky) space**
- 🔍 **Search for reconstruction lattice constants and rotations** 
  by calculating matching coefficients between experiment and theory

.. note::

   **xRHEED** is **not a GUI application**.  
   It is intended for use in **interactive environments** such as Jupyter notebooks, 
   where flexibility and scriptability are essential.
