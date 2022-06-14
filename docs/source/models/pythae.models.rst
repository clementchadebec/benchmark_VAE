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
    ~pythae.models.AE
    ~pythae.models.VAE
    ~pythae.models.BetaVAE
    ~pythae.models.VAE_LinNF
    ~pythae.models.VAE_IAF
    ~pythae.models.DisentangledBetaVAE
    ~pythae.models.FactorVAE
    ~pythae.models.BetaTCVAE
    ~pythae.models.IWAE
    ~pythae.models.MSSSIM_VAE
    ~pythae.models.WAE_MMD
    ~pythae.models.INFOVAE_MMD
    ~pythae.models.VAMP
    ~pythae.models.SVAE
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
    ...	    batch_size=200,
    ...	    steps_saving=None
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