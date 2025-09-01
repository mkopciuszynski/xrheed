Basic Usage
===========

Import xRHEED and load your data:

.. code-block:: python

    import xrheed
    from xrheed.io import load_data

    # Load RHEED image (replace with your file),
    rheed_image = load_data('rheed_image.raw', plugin="dsnp_arpes_raw")

    # Show a summary using ri accesory
    print(rheed_image.ri)

    # Plot the image
    rheed_image.ri.plot_image()

    # Get profile and print it's properties
    profile = rheed_image.ri.get_profile(center=(0, -5), width=40, height=4, plot_origin=True)
    print(profile.rp)

    # Plot profile
    profile.rp.plot_profile(
             transform_to_kx=True,
             normalize=True,
             color="black", linewidth=1.0
             )

For step-by-step guides, see the :doc:`example-notebooks` section.
