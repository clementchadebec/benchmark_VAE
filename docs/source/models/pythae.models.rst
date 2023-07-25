.. _pythae_models:

**********************************
Models
**********************************

This is the heart of Pythae. In here you will find implementations of **Autoencoder**-based models 
along with some **Normalizing Flows** used for improving Variational Inference in the VAE or 
sampling and **Neural Nets** to perform benchmark comparison.

.. toctree::
    :hidden:
    :maxdepth: 1

    autoencoders/models
    normalizing_flows/normalizing_flows
    nn/nn

Available Autoencoders
-------------------------------------------

.. autosummary::
    ~pythae.models.BaseAE
    ~pythae.models.AutoModel
    ~pythae.models.AE
    ~pythae.models.VAE
    ~pythae.models.BetaVAE
    ~pythae.models.VAE_LinNF
    ~pythae.models.VAE_IAF
    ~pythae.models.DisentangledBetaVAE
    ~pythae.models.FactorVAE
    ~pythae.models.BetaTCVAE
    ~pythae.models.IWAE
    ~pythae.models.CIWAE
    ~pythae.models.MIWAE
    ~pythae.models.PIWAE
    ~pythae.models.MSSSIM_VAE
    ~pythae.models.WAE_MMD
    ~pythae.models.INFOVAE_MMD
    ~pythae.models.VAMP
    ~pythae.models.SVAE
    ~pythae.models.PoincareVAE
    ~pythae.models.Adversarial_AE
    ~pythae.models.VAEGAN
    ~pythae.models.VQVAE
    ~pythae.models.HVAE
    ~pythae.models.RAE_GP
    ~pythae.models.RAE_L2
    ~pythae.models.RHVAE
    :nosignatures:

Available Normalizing Flows
-------------------------------------------

.. autosummary::
    ~pythae.models.normalizing_flows.BaseNF
    ~pythae.models.normalizing_flows.PlanarFlow
    ~pythae.models.normalizing_flows.RadialFlow
    ~pythae.models.normalizing_flows.MADE
    ~pythae.models.normalizing_flows.MAF
    ~pythae.models.normalizing_flows.IAF
    ~pythae.models.normalizing_flows.PixelCNN
    :nosignatures:


Basic Example
~~~~~~~~~~~~~~~

To launch a model training, you only need to set up your :class:`~pythae.trainers.BaseTrainerConfig`, 
:class:`~pythae.models.BaseAEConfig` build the model and trainer accordingly and then call a :class:`~pythae.pipelines.TrainingPipeline` 
instance. 

.. code-block::

    >>> from pythae.pipelines import TrainingPipeline
    >>> from pythae.models import VAE, VAEConfig
    >>> from pythae.trainers import BaseTrainerConfig

    >>> # Set up the training configuration
    >>> my_training_config = BaseTrainerConfig(
    ...	    output_dir='my_model',
    ...	    num_epochs=50,
    ...	    learning_rate=1e-3,
    ...	    per_device_train_batch_size=200,
    ...     per_device_eval_batch_size=200,
    ...	    steps_saving=None,
    ...     optimizer_cls="AdamW",
    ...	    optimizer_params={"weight_decay": 0.05, "betas": (0.91, 0.995)},
    ...	    scheduler_cls="ReduceLROnPlateau",
    ...	    scheduler_params={"patience": 5, "factor": 0.5}
    ... )
    >>> # Set up the model configuration 
    >>> my_vae_config = model_config = VAEConfig(
    ...	    input_dim=(1, 28, 28),
    ...	    latent_dim=10
    ... )
    >>> # Build the model
    >>> my_vae_model = VAE(
    ...	    model_config=my_vae_config
    ... )
    >>> # Build the Pipeline
    >>> pipeline = TrainingPipeline(
    ...     training_config=my_training_config,
    ...     model=my_vae_model
    ... )
    >>> # Launch the Pipeline
    >>> pipeline(
    ...	    train_data=your_train_data, # must be torch.Tensor or np.array 
    ...	    eval_data=your_eval_data # must be torch.Tensor or np.array
    ... )


.. _Distributed Training:

Distributed Training with Pythae
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
As of v0.1.0, Pythae now supports distributed training using PyTorch's  `DDP <https://pytorch.org/docs/stable/notes/ddp.html>`_. It allows you to train your favorite VAE faster and on larger dataset using multi-node and/or multi-gpu training.

To do so, you can built a python script that will then be launched by a launcher (such as `srun` on a cluster). The only thing that is needed in the script is to specify some elements relative to the environment (such as the number of nodes/gpus) directly in the training configuration as follows

.. code-block::

    >>> training_config = BaseTrainerConfig(
    ...     num_epochs=10,
    ...     learning_rate=1e-3,
    ...     per_device_train_batch_size=64,
    ...     per_device_eval_batch_size=64,
    ...     dist_backend="nccl", # distributed backend
    ...     world_size=8 # number of gpus to use (n_nodes x n_gpus_per_node),
    ...     rank=5 # process/gpu id,
    ...     local_rank=1 # node id,
    ...     master_addr="localhost" # master address,
    ...     master_port="1


See this `example script <https://github.com/clementchadebec/benchmark_VAE/tree/main/examples/scripts/distributed_training.py>`_ that defines a multi-gpu VQVAE training. Be carefull, the way the environnement (`world_size`, `rank` ...) may be specific to the cluster and launcher you use. 

Benchmark
############

Below are indicated the training times for a Vector Quantized VAE (VQ-VAE) with `Pythae` for 100 epochs 
on MNIST on V100 16GB GPU(s), for 50 epochs on `FFHQ <https://github.com/NVlabs/ffhq-dataset>`_ (1024x1024 images) 
and for 20 epochs on `ImageNet-1k <https://huggingface.co/datasets/imagenet-1k>`_ on V100 32GB GPU(s).

.. list-table:: Training times of a VQ-VAE with Pythae
   :widths: 25 25 25 25 25
   :header-rows: 1

   * - Dataset
     - Data type (train size)
     - 1 GPU
     - 4 GPUs
     - 2x4 GPUs
   * - MNIST
     - 28x28 images (50k)
     - 221.01s
     - 60.32s
     - 34.50s
   * - FFHQ
     - 1024x1024 RGB images (60k)
     - 19h 1min
     - 5h 6min
     - 2h 37min
   * - ImageNet-1k
     - 128x128 RGB images (~ 1.2M)
     - 6h 25min
     - 1h 41min
     - 51min 26s