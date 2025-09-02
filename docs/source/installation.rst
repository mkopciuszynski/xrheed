Installation
============

You can install **xRHEED** either with `pip` or with 
`uv <https://github.com/astral-sh/uv>`_, depending on your workflow.

---

Using pip (recommended for development)
---------------------------------------

For a development setup with editable installation:

.. code-block:: bash

   git clone https://github.com/mkopciuszynski/xrheed
   cd xrheed
   pip install -e .

This allows you to make changes to the source code and immediately test them.

---

Using uv (modern package & environment manager)
-----------------------------------------------

`uv` simplifies dependency management and virtual environments.

1. Install `uv` following the 
   `official guide <https://docs.astral.sh/uv/guides/projects/>`_.
2. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/mkopciuszynski/xrheed
      cd xrheed

3. Create and activate a virtual environment (command depends on your shell).
4. Synchronize dependencies:

   .. code-block:: bash

      uv sync

---

Basic dependencies
------------------

- `xarray` for labeled array structures
- `numpy` and `scipy` for numerical operations
- `matplotlib` for plotting

Optional packages may be required for plugins or specialized workflows.
