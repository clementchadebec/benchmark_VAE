"""Implementation of an Adversarial Autoencoder model as proposed in 
(https://arxiv.org/abs/1511.05644). This model tries to make the posterior distribution match 
the prior using adversarial training.

Available samplers
-------------------

.. autosummary::
    ~pythae.samplers.NormalSampler
    ~pythae.samplers.GaussianMixtureSampler
    ~pythae.samplers.TwoStageVAESampler
    ~pythae.samplers.MAFSampler
    ~pythae.samplers.IAFSampler
    :nosignatures:


"""

from .adversarial_ae_config import Adversarial_AE_Config
from .adversarial_ae_model import Adversarial_AE

__all__ = ["Adversarial_AE", "Adversarial_AE_Config"]
