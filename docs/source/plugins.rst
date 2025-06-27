Plugins for Data Loading
=======================

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
  - `x` (horizontal) and `y` (vertical) coordinates, both in millimeters.
  - Coordinates sorted so that the image is oriented with the shadow edge at the top (i.e., the image is "facing down").
  - The `y` values covering the image should be negative (i.e., the top of the image has the most negative `y`).

**Example plugin attributes:**

.. code-block:: python

    ATTRS = {
        "plugin": "UMCS DSNP ARPES Raw",
        "screen_sample_distance": 309.2,
        "screen_scale": 9.6,
        "screen_center_x": 72.0,  # horizontal center of an image
        "screen_center_y": 15.0,  # shadow edge position
        "beam_energy": 18.6 * 1000,
    }

**Returned DataArray:**

- The data should be an `xarray.DataArray` with coordinates:
    - `x`: horizontal axis, in mm
    - `y`: vertical axis, in mm
- The image should be oriented so the shadow edge is at the top, what means that we get negative values for the image area.

This ensures that all loaded images are consistent and ready for further analysis and visualization in xRHEED.

