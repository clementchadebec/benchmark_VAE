**********************************
Benchmarks
**********************************

.. toctree::
   :maxdepth: 1

   mnist/pythae_benchmarks_nn_mnist
   cifar/pythae_benchmarks_nn_cifar
   celeba/pythae_benchmarks_nn_celeba

.. automodule::
   pythae.models.nn.benchmarks


MNIST
*********

.. automodule::
   pythae.models.nn.benchmarks.mnist

ConvNets
---------------

.. autosummary::
   ~pythae.models.nn.benchmarks.mnist.Encoder_Conv_AE_MNIST
   ~pythae.models.nn.benchmarks.mnist.Encoder_Conv_VAE_MNIST
   ~pythae.models.nn.benchmarks.mnist.Encoder_Conv_SVAE_MNIST
   ~pythae.models.nn.benchmarks.mnist.Decoder_Conv_AE_MNIST
   ~pythae.models.nn.benchmarks.mnist.Discriminator_Conv_MNIST
   :nosignatures:


ResNets
---------------

.. autosummary::
   ~pythae.models.nn.benchmarks.mnist.Encoder_ResNet_AE_MNIST
   ~pythae.models.nn.benchmarks.mnist.Encoder_ResNet_VAE_MNIST
   ~pythae.models.nn.benchmarks.mnist.Encoder_ResNet_SVAE_MNIST
   ~pythae.models.nn.benchmarks.mnist.Decoder_ResNet_AE_MNIST
   :nosignatures:

CIFAR
*********

.. automodule::
   pythae.models.nn.benchmarks.cifar

ConvNets
---------------

.. autosummary::
   ~pythae.models.nn.benchmarks.cifar.Encoder_Conv_AE_CIFAR
   ~pythae.models.nn.benchmarks.cifar.Encoder_Conv_VAE_CIFAR
   ~pythae.models.nn.benchmarks.cifar.Encoder_Conv_SVAE_CIFAR
   ~pythae.models.nn.benchmarks.cifar.Decoder_Conv_AE_CIFAR
   ~pythae.models.nn.benchmarks.cifar.Discriminator_Conv_CIFAR
   :nosignatures:

ResNets
---------------

.. autosummary::
   ~pythae.models.nn.benchmarks.cifar.Encoder_ResNet_AE_CIFAR
   ~pythae.models.nn.benchmarks.cifar.Encoder_ResNet_VAE_CIFAR
   ~pythae.models.nn.benchmarks.cifar.Encoder_ResNet_SVAE_CIFAR
   ~pythae.models.nn.benchmarks.cifar.Decoder_ResNet_AE_CIFAR
   :nosignatures:

CELEBA-64
*********

.. automodule::
   pythae.models.nn.benchmarks.celeba

ConvNets
---------------

.. autosummary::
   ~pythae.models.nn.benchmarks.celeba.Encoder_Conv_AE_CELEBA
   ~pythae.models.nn.benchmarks.celeba.Encoder_Conv_VAE_CELEBA
   ~pythae.models.nn.benchmarks.celeba.Encoder_Conv_SVAE_CELEBA
   ~pythae.models.nn.benchmarks.celeba.Decoder_Conv_AE_CELEBA
   ~pythae.models.nn.benchmarks.celeba.Discriminator_Conv_CELEBA
   :nosignatures:

ConvNets
---------------

.. autosummary::
   ~pythae.models.nn.benchmarks.celeba.Encoder_ResNet_AE_CELEBA
   ~pythae.models.nn.benchmarks.celeba.Encoder_ResNet_VAE_CELEBA
   ~pythae.models.nn.benchmarks.celeba.Encoder_ResNet_SVAE_CELEBA
   ~pythae.models.nn.benchmarks.celeba.Decoder_ResNet_AE_CELEBA
   :nosignatures:

.. note::

   In case you want to provide your own neural architecture, make sure you make them inherit from these classes.
   See tutorials.