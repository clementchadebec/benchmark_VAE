.. _pythae_models:

**********************************
Models
**********************************

This is the heart of Pythae. In here you will find implementations of **Autoencoder**-based models 
along with some **Normalizing Flows** used from improving Variational Inference in the VAE os 
sampling and **Neural Nets** to perform benchmark comparison.

.. toctree::
    :hidden:
    :maxdepth: 1

    autoencoders/models
    normalizing_flows/normalizing_flows
    nn/nn
    :nosignatures:

Available Autoencoders
-----------------

.. autosummary::
    ~pythae.models.BaseAE
    ~pythae.models.AE
    ~pythae.models.VAE
    ~pythae.models.BetaVAE
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
-----------------

.. autosummary::
    ~pythae.models.normalizing_flows.BaseNF
    ~pythae.models.normalizing_flows.MADE
    ~pythae.models.normalizing_flows.MAF
    ~pythae.models.normalizing_flows.IAF
    :nosignatures:
