"""
Implementation of the Inverse Autoregressive Flows (IAF) proposed
in (https://arxiv.org/abs/1606.04934)

All the codes are inspired from 
- (https://github.com/kamenbliznashki/normalizing_flows)
- (https://github.com/karpathy/pytorch-normalizing-flows)
- (https://github.com/ikostrikov/pytorch-flows)
"""

from .iaf_config import IAFConfig
from .iaf_model import IAF

__all__ = [
    "IAF",
    "IAFConfig"
]