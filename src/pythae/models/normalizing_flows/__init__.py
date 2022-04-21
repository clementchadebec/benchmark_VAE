"""
In this module are implemented so normalizing flows that can be used within the vae model or for 
sampling.

By convention, each implemented model is stored in a folder located in 
:class:`pythae.models.normalizing_flows` and named likewise the model. The following modules can be 
found in this folder:

- | *modelname_config.py*: Contains a :class:`ModelNameConfig` instance inheriting
    from :class:`~pythae.models.normalizing_flows.BaseNF`. 
- | *modelname_model.py*: An implementation of the model inheriting either from
    :class:`~pythae.models.normalizing_flows.BaseNFConfig` 
- *modelname_utils.py* (optional): A module where utils methods are stored.

All the codes are inspired from:

- | https://github.com/kamenbliznashki/normalizing_flows
- | https://github.com/karpathy/pytorch-normalizing-flows
- | https://github.com/ikostrikov/pytorch-flows
"""

from .base import BaseNF, BaseNFConfig, NFModel
from .iaf import IAF, IAFConfig
from .made import MADE, MADEConfig
from .maf import MAF, MAFConfig
from .pixelcnn import PixelCNN, PixelCNNConfig
from .planar_flow import PlanarFlow, PlanarFlowConfig
from .radial_flow import RadialFlow, RadialFlowConfig

__all__ = [
    "BaseNF",
    "BaseNFConfig",
    "NFModel",
    "MADE",
    "MADEConfig",
    "MAF",
    "MAFConfig",
    "IAF",
    "IAFConfig",
    "PlanarFlow",
    "PlanarFlowConfig",
    "RadialFlow",
    "RadialFlowConfig",
    "PixelCNN",
    "PixelCNNConfig",
]
