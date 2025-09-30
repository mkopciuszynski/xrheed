# Changelog

<a name="1.3.0"></a>
## [1.3.0] – 2025-09-30

- Expose `load_data` and `load_data_manual` at the package root (`xrheed.load_data`) for simpler imports
- Update documentation and examples to reflect the new recommended loading approach
- Adopt setuptools_scm for dynamic versioning from git tags
- Expose `__version__` in package root
- Auto-update CITATION.cff version during release
- Simplify workflow for building and publishing to PyPI

<a name="1.1.2"></a>
## [1.1.2] – 2025-09-19

- Fix quick usage in README (use xrheed.loaders)
- Minor fixes in example notebooks
- Update citation

<a name="1.1.1"></a>
## [1.1.1] – 2025-09-15

- **Add** `load_data_manual()` to load RHEED images when no plugin is available.
- **Refactor** `io.py` → `loaders.py`  
  - Breaking change: code using `from xrheed.io import ...` must now use `from xrheed.loaders import ...`  
  - Resolves the naming conflict with Python's standard library `io` and clarifies the module’s purpose (data loading only).

- **Documentation fixes and updates:**  
  - Updated example notebook sections.  
  - Updated badges.
  - Updated plugin and xarray_accessors references in the API documentation.


<a name="1.0.0"></a>
## [1.0.0] – 2025-09-11

**First stable release of xRHEED.**
- Project is now officially public and stable.
- Automatically uploaded to PyPI.
- Documentation hosted on Read the Docs.
- Zenodo DOI assigned: [10.5281/zenodo.17099752](https://doi.org/10.5281/zenodo.17099752)
- No significant changes to core code since v0.5.x.

<a name="0.5.0"></a>
## [0.5.0] – 2025-09-11

- PyPI-ready release with automatic publishing via GitHub Actions.
- Full Sphinx documentation with myst-nb and API reference.
- Added `.readthedocs.yaml` for Read the Docs builds.
- Updated `docs/source/conf.py` for autodoc, autosummary, and notebook support.
- Updated CI workflow (`ci.yml`) to build docs, run tests, lint code, and publish releases.
- Added `CITATION.cff` for formal citation and Zenodo DOI integration.


<a name="0.4.0"></a>
## [0.4.0] – 2025-09-11
- Refactored LoadRheedBase to use an abstract base class with __init_subclass__ validation.
- Enforced presence of required ATTRS keys in all plugins.
- Added dsnp_arpes_bmp plugin to support BMP image loading via Pillow.
- Implemented automatic plugin discovery in xrheed/__init__.py.
- Refactored test suite to dynamically validate all supported plugins and file types.
- Updated CONTRIBUTING.md


<a name="0.3.0"></a>
## [0.3.0] – 2025-09-09
- New argument: show_specular_spot available in plot_image
- New example notebook showing how to search for lattice constant and azimuthal orientation for a given RHEED data.
- A major update in the Ewald class including:
    - Ewald matching functions rewritten
    - Added decorator that saves the matching results to cache dill files
    - New constants
    - Type hints
    - Better docstring


<a name="0.2.0"></a>
## [0.2.0] – 2025-09-04
- A major update in the documentation
- New example images 
- Polished and improved markdowns in jupyter notebooks
- Docstring added, and API ready
- Profile methods used for transformation now use a proper geometry sx -> ky


<a name="0.1.0"></a>
## [0.1.0] – 2025-08-29
- First working release with core functionality
- Load and preprocess RHEED images
- Generate and analyze intensity profiles
- Overlay predicted diffraction spot positions (kinematic theory & Ewald construction)
- Documentation with few example notebooks