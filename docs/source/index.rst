.. pythae documentation master file, created by
   sphinx-quickstart on Wed Jun  2 16:47:13 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

**********************************
Welcome to pythae's documentation!
**********************************

This library aims at gathering some of the common (Variational) Autoencoders implementations so that
we can conduct benchmark analysis and reproducible research!

**News** 📢

As of v0.1.0, `Pythae` now supports distributed training using PyTorch's `DDP <https://pytorch.org/docs/stable/notes/ddp.html>`_). You can now train your favorite VAE faster and on larger datasets, still with a few lines of code.
See :ref:`Distributed Training`.

.. toctree::
   :maxdepth: 1
   :caption: Pythae
   :titlesonly:

   models/pythae.models
   samplers/pythae.samplers
   trainers/pythae.trainer
   pipelines/pythae.pipelines

Setup
~~~~~~~~~~~~~

To install the latest stable release of this library run the following using ``pip``

.. code-block:: bash

   $ pip install pythae

To install the latest version of this library run the following using ``pip``

.. code-block:: bash

   $ pip install git+https://github.com/clementchadebec/benchmark_VAE.git 

or alternatively you can clone the github repo to access to tests, tutorials and scripts.

.. code-block:: bash

   $ git clone https://github.com/clementchadebec/benchmark_VAE.git

and install the library

.. code-block:: bash

   $ cd benchmark_VAE
   $ pip install -e .

If you clone the pythae's repository you will access to  the following:

- ``docs``: The folder in which the documentation can be retrieved.
- ``tests``: pythae's unit-testing using pytest.
- ``examples``: A list of ``ipynb`` tutorials and script describing the main functionalities of pythae.
- ``src/pythae``: The main library which can be installed with ``pip``.