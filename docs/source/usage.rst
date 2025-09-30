Quick Usage
===========

Below is a minimal example to demonstrate the workflow with **xRHEED**.

.. code-block:: python

   import matplotlib.pyplot as plt
   import xrheed

   # Load a RHEED image
   rheed_image = xrheed.load_data("rheed_image.raw", plugin="dsnp_arpes_raw")

   # Show the image with auto-adjusted intensity levels
   rheed_image.ri.plot_image(auto_levels=2.0)
   plt.show()

   # Extract an intensity profile
   profile = rheed_image.ri.get_profile(
       center=(0, -5), 
       width=40, 
       height=4, 
       plot_origin=True
   )


Working with Profiles
---------------------

Intensity profiles are useful for tracking diffraction streaks or 
quantifying reconstruction features.  
They can be directly plotted or exported for further analysis.

Example:

.. code-block:: python

   fig, ax = plt.subplots()
   ax.plot(profile.coords["x"], profile.values)
   ax.set_xlabel("Reciprocal coordinate (1/Å)")
   ax.set_ylabel("Intensity")
   plt.show()


Next Steps
----------

- Explore the :doc:`geometry` section for details on RHEED geometry.
- See :doc:`notebooks/getting_started` and other :ref:`example-notebooks` for interactive demonstrations.
- Dive into the :doc:`xrheed` API reference for function-level details.
