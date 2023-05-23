"""
Implementation of the Inverse Autoregressive Flows (IAF) proposed
in (https://arxiv.org/abs/1606.04934).
"""

from .iaf_config import IAFConfig
from .iaf_model import IAF

__all__ = ["IAF", "IAFConfig"]
