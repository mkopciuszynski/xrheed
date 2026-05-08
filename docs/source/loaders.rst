Data Loading
============

xRHEED provides a unified and robust data-loading interface for RHEED images.
Images can be loaded either via **plugins** (recommended) or via **manual loading**
(for quick tests or getting started). In both cases, the result is an
``xarray.DataArray`` following the same structural conventions.

Using Plugins
-------------

xRHEED uses a flexible plugin system to load RHEED images and attach experiment-
specific geometry and metadata.

A plugin represents knowledge about a particular file format or instrument.
It is responsible for loading image data and describing *what is known* about the experiment.

A plugin should:

- **Load a specific data format** (e.g. ``.raw``, ``.png``, ``.bmp``).
- **Declare instrument and geometry defaults** via an ``ATTRS`` dictionary.
- **Return a canonical ``xarray.DataArray``** using the provided helper.

### Plugin ``ATTRS``

The ``ATTRS`` dictionary defines defaults describing the experimental setup.
Typical keys include:

- ``plugin``: Human-readable plugin name.
- ``screen_sample_distance``: Distance from sample to screen [mm].
- ``screen_scale``: Pixel-to-mm scaling factor.
- ``screen_center_sx_px``: Horizontal center of the image [px].
  If not provided, the image midpoint is used.
- ``screen_center_sy_px``: Vertical position of the shadow edge [px].
  This is instrument-specific and should normally be provided.
- ``beam_energy``: Electron beam energy [eV].

Acquisition parameters such as ``alpha`` (azimuthal angle) and ``beta`` (incident angle),
if read from file metadata by a plugin, may be attached to ``ATTRS``, 
but they are promoted to coordinates exclusively by the loader.

Plugins **do not** define scan dimensions, stack images, or decide whether
metadata should become coordinates. That responsibility belongs to the loader.

Helper Function
^^^^^^^^^^^^^^^

Plugins typically use the helper method ``dataarray_from_image()``, which:

- resolves geometry (including screen centers),
- creates ``sx`` and ``sy`` coordinates in millimeters,
- attaches file provenance metadata.

Manual Data Loading
-------------------

While writing and using a plugin is recommended, xRHEED also supports **manual
single-image loading** for quick inspection or prototyping.

Manual loading uses the same ``load_data()`` function, but without specifying
a plugin.

In this mode, only a **single image** may be loaded.
The user must provide the essential calibration parameters explicitly.

### Required parameters

- ``screen_sample_distance``: Distance from sample to screen [mm].
- ``screen_scale``: Pixel-to-mm scaling factor.
- ``beam_energy``: Beam energy [eV].

### Optional parameters

- ``screen_center_sx_px``: Horizontal image center [px].
- ``screen_center_sy_px``: Shadow edge position [px].
  If omitted, the top of the image (0 px) is used with a warning.
- ``alpha`` / ``beta``: Acquisition angles, if known.

Example:

.. code-block:: python

    import xrheed

    rheed_image = xrheed.load_data(
        "example.bmp",
        screen_scale=9.04,
        screen_sample_distance=309.2,
        beam_energy=18_600,
    )


Angles and Coordinates
----------------------

Acquisition angles (``alpha`` and ``beta``) are treated as **coordinates**,
not attributes.

- If angles are provided per image and vary across a stack, they are
  promoted to coordinates by the loader.
- If angles are missing:
  - single-images returns ``None``, angles might be added via accessors,
  - stacked data requires explicit coordinates and will raise an error.

Returned DataArray
------------------

All loading methods return an ``xarray.DataArray`` with:

- dimensions ``("sy", "sx")`` for single images,
- coordinates:
  - ``sx``: horizontal axis [mm],
  - ``sy``: vertical axis [mm],
- the image oriented so the shadow edge is at the top.

This ensures consistent downstream analysis, visualization, and stacking
within xRHEED.


Example Plugin
--------------

A complete, minimal reference plugin demonstrating best practices for plugin-based
loading is provided with the source code:

.. literalinclude:: ../../../src/xrheed/plugins/example_plugin_bmp.py
   :language: python