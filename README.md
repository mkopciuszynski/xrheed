# xRHEED

📡 An **xarray-based toolkit** for RHEED image analysis.

---

[![CI](https://github.com/mkopciuszynski/xrheed/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/mkopciuszynski/xrheed/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/xrheed/badge/?version=latest)](https://xrheed.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://img.shields.io/pypi/v/xrheed.svg)](https://pypi.org/project/xrheed/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Linter: ruff](https://img.shields.io/badge/linter-ruff-46a2f1.svg?logo=ruff)](https://github.com/astral-sh/ruff)
[![Package manager: uv](https://img.shields.io/badge/packaging-uv-blue)](https://github.com/astral-sh/uv)
[![DOI](https://zenodo.org/badge/963155496.svg)](https://doi.org/10.5281/zenodo.17099751)

---

## 🔬 What is RHEED?

**Reflection High-Energy Electron Diffraction (RHEED)** is an experimental technique used to monitor and control the quality of crystal surfaces.  
A high-energy electron beam (∼20 keV) strikes the surface at a grazing angle (< 5°), making the method highly **surface-sensitive** and probing only a few atomic layers.

---

## 🎯 Project Goals

**xRHEED** provides a flexible and extensible **Python toolkit** for RHEED image analysis:

- 🖼️ Load and preprocess RHEED images  
- 📈 Generate and analyze intensity profiles  
- ✨ Overlay predicted diffraction spot positions (kinematic theory & Ewald construction)  
- 🔄 Transform RHEED images into kx–ky space  
- 🔍 Search for reconstruction lattice constants and rotations by calculating the matching coefficient between predicted and experimental data  

👉 **Note:** xRHEED is **not a GUI application**. It is designed as an **xarray accessory library**, intended for use in **interactive environments** such as Jupyter notebooks.

---

## Installation

### Using PyPI

```bash
pip install xrheed
```

### Using pip (editable install for development)

```bash
git clone https://github.com/mkopciuszynski/xrheed
cd xrheed
pip install -e .
```

### Using uv (with virtual environment)

1. Install [`uv`](https://docs.astral.sh/uv/guides/projects/).
2. Clone the repository:
```bash
git clone https://github.com/mkopciuszynski/xrheed
cd xrheed
```
3. Create and activate a virtual environment.
4. Sync dependencies:
```bash
uv sync
```
---

## 🚀 Quick Usage

```python
import matplotlib.pyplot as plt
import xrheed
from xrheed.io import load_data

# Load a RHEED image
rheed_image = load_data("rheed_image.raw", plugin="dsnp_arpes_raw")

# Show image with auto-adjusted levels
rheed_image.ri.plot_image(auto_levels=2.0)
plt.show()

# Get intensity profile and plot its origin
profile = rheed_image.ri.get_profile(center=(0, -5), width=40, height=4,
                                     plot_origin=True)
```

---

## 📖 Citation

If you use **xRHEED** in your research, please cite it:

> Kopciuszynski, M. [ORCID](https://orcid.org/0000-0001-7360-6829) (2025). *xRHEED: An xarray-based toolkit for RHEED image analysis*.  
> GitHub. https://github.com/mkopciuszynski/xrheed  
> DOI: [10.5281/zenodo.17099752](https://doi.org/10.5281/zenodo.17099752)

---

📚 👉 See the [full documentation](https://xrheed.readthedocs.io/en/latest/) for tutorials and advanced examples.
