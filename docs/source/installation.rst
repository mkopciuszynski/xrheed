Installation
============

You can install **xRHEED** in several ways:

Using pip (from PyPI)
---------------------

The easiest way to install the latest release from PyPI:

.. code-block:: bash

   pip install xrheed

Using pip (editable install for development)
--------------------------------------------

If you want to work with the development version:

.. code-block:: bash

   git clone https://github.com/mkopciuszynski/xrheed
   cd xrheed
   pip install -e .

Using uv (with a virtual environment)
-------------------------------------

1. Install `uv <https://docs.astral.sh/uv/guides/projects/>`_.
2. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/mkopciuszynski/xrheed
      cd xrheed

3. Create and activate a virtual environment (depending on your shell: bash, zsh, fish, PowerShell).
4. Sync dependencies:

   .. code-block:: bash

      uv sync

Dependencies
------------

**xRHEED** builds on the scientific Python ecosystem and requires:

- `xarray`
- `numpy`
- `scipy`
- `matplotlib`

For working with example notebooks and documentation, `jupyter` is also recommended.