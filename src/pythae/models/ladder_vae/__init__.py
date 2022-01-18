"""This module is the implementation of the Ladder VAE proposed in
(https://arxiv.org/abs/1602.02282v3).


Available samplers
-------------------

"""

from .ladder_vae_config import LadderVAEConfig
from .ladder_vae_model import LadderVAE

__all__ = ["LadderVAE", "LadderVAEConfig"]
