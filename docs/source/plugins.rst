Plugins for Data Loading
========================

xRHEED uses a plugin system to load RHEED images and provide geometry information for each experiment.

A plugin should:

- **Load a particular data format** (e.g., `.raw`, `.png`, `.bmp`).
- **Include RHEED geometry** by defining an `ATTRS` dictionary with keys such as:

  - `"plugin"`: Name of the plugin.
  - `"screen_sample_distance"`: Distance from sample to screen (in mm).
  - `"screen_scale"`: Pixel-to-mm scaling factor.
  - `"screen_center_x"`: Horizontal center of the image (in pixels).
  - `"screen_center_y"`: Vertical position of the shadow edge (in pixels).
  - `"beam_energy"`: Electron beam energy (in eV).

  The values of the screen center are only approximate since they may change from one experiment to other.

- **Return an `xarray.DataArray`** with:
  - `sx` (horizontal) and `sy` (vertical) coordinates, both in millimeters.
  - Coordinates sorted so that the image is oriented with the shadow edge at the top (i.e., the image is "facing down").
  - The `sy` values covering the image should be negative (i.e., the top of the image has the most negative `sy`).

**Example plugin attributes:**

.. code-block:: python

    ATTRS = {
        "plugin": "UMCS DSNP ARPES raw",
        "screen_sample_distance": 309.2,  # mm
        "screen_scale": 9.04,  # pixels per mm
        "screen_center_sx_px": 740,  # horizontal center of an image in px
        "screen_center_sy_px": 155,  # shadow edge position in px
        "beam_energy": 18.6 * 1000,  # eV
        "alpha": 0.0,  # azimuthal angle
        "beta": 2.0,   # incident angle
    }


**Returned DataArray:**

- The data should be an `xarray.DataArray` with coordinates:
    - `sx`: horizontal axis, in mm
    - `sy`: vertical axis, in mm
- The image should be oriented so the shadow edge is at the top, what means that we get negative values for the image area.

This ensures that all loaded images are consistent and ready for further analysis and visualization in xRHEED.
