"""

**Abstract class**

Base module for Normalizing Flows implementation"""

from .base_nf_config import BaseNFConfig
from .base_nf_model import BaseNF

__all__ = [
    "BaseNF",
    "BaseNFConfig"
]