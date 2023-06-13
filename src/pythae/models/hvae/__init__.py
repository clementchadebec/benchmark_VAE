"""This module is the implementation of the Hamiltonian VAE proposed in
(https://arxiv.org/abs/1805.11328).
This model combines Hamiltonian Monte Carlo smapling and normalizing flows together to improve the 
true posterior estimate within the VAE framework.

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

from .hvae_config import HVAEConfig
from .hvae_model import HVAE

__all__ = ["HVAE", "HVAEConfig"]
