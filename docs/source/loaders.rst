Data Loading
============

Using Plugins
-------------

xRHEED uses a plugin system to load RHEED images and provide geometry information for each experiment.

A plugin should:

- **Load a particular data format** (e.g., ``.raw``, ``.png``, ``.bmp``).  
- **Include RHEED geometry** by defining an ``ATTRS`` dictionary with keys such as:

  - ``plugin``: Name of the plugin.  
  - ``screen_sample_distance``: Distance from sample to screen (in mm).  
  - ``screen_scale``: Pixel-to-mm scaling factor.  
  - ``screen_center_sx_px``: Horizontal center of the image (in pixels).  
  - ``screen_center_sy_px``: Vertical position of the shadow edge (in pixels).  
  - ``beam_energy``: Electron beam energy (in eV).  

  .. note::
     The screen center values are approximate and may vary between experiments.

- **Return an ``xarray.DataArray``** with:
  - ``sx`` (horizontal) and ``sy`` (vertical) coordinates, both in millimeters.  
  - Coordinates sorted so that the image is oriented with the shadow edge at the top (i.e., the image is "facing down").  
  - The ``sy`` values covering the image should be negative at the top of the image.

Example plugin attributes:

.. code-block:: python

    ATTRS = {
        "plugin": "UMCS DSNP ARPES raw",
        "screen_sample_distance": 309.2,  # mm
        "screen_scale": 9.04,  # pixels per mm
        "screen_center_sx_px": 740,  # horizontal center of an image in px
        "screen_center_sy_px": 155,  # shadow edge position in px
        "beam_energy": 18_600,  # eV
        "alpha": 0.0,  # default azimuthal angle
        "beta": 2.0,   # default incident angle
    }



Manual
------

While writing and using a dedicated plugin is the recommended approach, it is also possible to load RHEED images manually (BMP, PNG, TIFF, or other formats) using the ``load_data_manual()`` function.

In this mode, the user must provide the essential calibration parameters directly:

- ``screen_sample_distance``: Distance from sample to screen [mm].  
- ``screen_scale``: Pixel-to-mm scaling factor.
- ``beam_energy``: Beam energy [eV].

Optional parameters include ``screen_center_sx_px``, ``screen_center_sy_px``, ``alpha``, and ``beta``.  
The resulting ``xarray.DataArray`` follows the same conventions as plugin-loaded images (coordinates in mm, shadow edge at the top).

Example usage:

.. code-block:: python

    import xrheed

    rheed_image = xrheed.load_data_manual(
        "example.bmp",
        screen_sample_distance=309.2,
        screen_scale=9.04,
        beam_energy=18_600,  # eV
    )


This ensures that all loaded images, whether via a plugin or manually, are consistent and ready for further analysis and visualization in xRHEED.

Returned DataArray
------------------

- The data should be an ``xarray.DataArray`` with coordinates:
    - ``sx``: horizontal axis, in mm  
    - ``sy``: vertical axis, in mm  
- The image should be oriented so the shadow edge is at the top (negative ``sy`` values).
