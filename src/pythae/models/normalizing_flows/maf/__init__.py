"""
Implementation of the Masked Autoregressive Flow (MAF) proposed
in (https://arxiv.org/abs/1502.03509)
"""

from .maf_config import MAFConfig
from .maf_model import MAF

__all__ = ["MAF", "MAFConfig"]
