"""

**Abstract class**

Base module for Normalizing Flows implementation"""

from .base_nf_config import BaseNFConfig
from .base_nf_model import BaseNF, NFModel

__all__ = ["NFModel", "BaseNF", "BaseNFConfig"]
