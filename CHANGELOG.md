# Changelog

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