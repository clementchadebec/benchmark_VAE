"""Implementation of an Adversarial Autoencoder model as proposed in 
(https://arxiv.org/pdf/1511.05644.pdf)

Available samplers
-------------------

.. autosummary::
    ~pythae.samplers.NormalSampler
    ~pythae.samplers.GaussianMixtureSampler
    :nosignatures:

"""

from .adversarial_ae_model import Adversarial_AE
from .adversarial_ae_config import Adversarial_AE_Config

__all__ = ["Adversarial_AE", "Adversarial_AE_Config"]
