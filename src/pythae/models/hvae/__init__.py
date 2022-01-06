"""This module is the implementation of the Hamiltonian VAE proposed in
(https://proceedings.neurips.cc/paper/2018/file/3202111cf90e7c816a472aaceb72b0df-Paper.pdf).
This model combines Hamiltonian Monte Carlo smapling and normalizing flows together to improve the 
true posterior estimate within the VAE framework.

Available samplers
-------------------

.. autosummary::
    ~pythae.samplers.NormalSampler
    ~pythae.samplers.GaussianMixtureSampler
    ~pythae.samplers.TwoStageVAESampler
    :nosignatures:
"""

from .hvae_config import HVAEConfig
from .hvae_model import HVAE

__all__ = ["HVAE", "HVAEConfig"]
