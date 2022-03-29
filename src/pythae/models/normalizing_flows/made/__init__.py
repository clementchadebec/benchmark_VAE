"""
Implementation of the Masked Autoencoder model (MADE) proposed
in (https://arxiv.org/abs/1502.03509)

All the codes are inspired from 
- (https://github.com/kamenbliznashki/normalizing_flows)
- (https://github.com/karpathy/pytorch-normalizing-flows)
- (https://github.com/ikostrikov/pytorch-flows)
"""

from .made_config import MADEConfig
from .made_model import MADE

__all__ = ["MADE", "MADEConfig"]
