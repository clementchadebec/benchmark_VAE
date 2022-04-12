"""Proposed residual neural nets architectures suited for MNIST"""

from turtle import forward
from typing import List

import numpy as np
import torch
import torch.nn as nn
from pythae.models.nn import BaseDecoder, BaseDiscriminator, BaseEncoder

from ....base import BaseAEConfig
from ....base.base_utils import ModelOutput
from ...base_architectures import BaseDecoder, BaseEncoder


class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        nn.Module.__init__(self)

        self.conv_block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=1)
        )

    def forward(self, x: torch.tensor) -> torch.Tensor:
        