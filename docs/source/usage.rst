Basic Usage
===========

Import xRHEED and load your data:

.. code-block:: python

    import xrheed

    # Load example RHEED data (replace with your file)
    data = xrheed.load_rheed('your_data_file.raw')

    # Show a summary
    print(data)

    # Plot data
    data.plot()

For step-by-step guides, see the example notebooks:

.. toctree::
   :maxdepth: 1

   examples/getting_started.ipynb