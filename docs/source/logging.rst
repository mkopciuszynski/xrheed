Logging
=======

``xrheed`` records key analysis steps and parameter updates using Python's :mod:`logging` module.  
Logs help track data processing and improve reproducibility during RHEED analysis.

Example
-------

Below is a minimal example showing how to enable logging and view messages directly in a notebook or script:

.. code-block:: python

   import matplotlib.pyplot as plt
   import logging

   logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

   import xrheed

   image_path = "example_data/Si_111_7x7_112_phi_00.raw"
   rheed_image = xrheed.load_data(image_path, plugin="dsnp_arpes_raw")

   rheed_image.ri.set_center_auto()
   rheed_image.ri.rotate(0.5)

   rheed_image.ri.plot_image(auto_levels=0.1)
   plt.show()

Example output:

.. code-block:: text

   INFO: Loading file 'example_data\\Si_111_7x7_112_phi_00.raw' using plugin 'dsnp_arpes_raw'
   INFO: Horizontal center estimated at -0.2212
   INFO: Vertical center estimated at 1.6406
   INFO: Applied automatic centering: center_x=-0.2212, center_y=1.6406
   INFO: Rotation applied: angle=0.5000 degrees
