**********************************
Benchmarks
**********************************

.. toctree::
   :maxdepth: 1

   pythae_benchmarks_nn_mnist
   pythae_benchmarks_nn_cifar
   pythae_benchmarks_nn_celeba

.. automodule::
   pythae.models.nn.benchmarks


MNIST
*********

.. automodule::
   pythae.models.nn.benchmarks.mnist

.. autosummary::
   ~pythae.models.nn.benchmarks.mnist.Encoder_AE_MNIST
   ~pythae.models.nn.benchmarks.mnist.Encoder_VAE_MNIST
   ~pythae.models.nn.benchmarks.mnist.Decoder_AE_MNIST
   :nosignatures:

CIFAR
*********

.. automodule::
   pythae.models.nn.benchmarks.cifar

.. autosummary::
   ~pythae.models.nn.benchmarks.cifar.Encoder_AE_MNIST
   ~pythae.models.nn.benchmarks.cifar.Encoder_VAE_MNIST
   ~pythae.models.nn.benchmarks.cifar.Decoder_AE_MNIST
   :nosignatures:

CELEBA-64
*********

.. automodule::
   pythae.models.nn.benchmarks.celeba

.. autosummary::
   ~pythae.models.nn.benchmarks.celeba.Encoder_AE_MNIST
   ~pythae.models.nn.benchmarks.celeba.Encoder_VAE_MNIST
   ~pythae.models.nn.benchmarks.celeba.Decoder_AE_MNIST
   :nosignatures:

.. note::

   In case you want to provide your own neural architecture, make sure you make them inherit from these classes. See :ref:`making-your-own-vae`.