"""
Implementation of Planar Flows (PlanarFlow) proposed
in (https://arxiv.org/abs/1505.05770)
"""

from .planar_flow_config import PlanarFlowConfig
from .planar_flow_model import PlanarFlow

__all__ = ["PlanarFlow", "PlanarFlowConfig"]
