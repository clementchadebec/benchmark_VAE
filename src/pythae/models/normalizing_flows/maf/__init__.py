"""
Implementation of the Masked Autoregressive Flow (MAF) proposed
in (https://arxiv.org/abs/1502.03509)

All the codes are inspired from 
- (https://github.com/kamenbliznashki/normalizing_flows)
- (https://github.com/karpathy/pytorch-normalizing-flows)
- (https://github.com/ikostrikov/pytorch-flows)
"""

from .maf_config import MAFConfig
from .maf_model import MAF

__all__ = [
    "MAF",
    "MAFConfig"
]