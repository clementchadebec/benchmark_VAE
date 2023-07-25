from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedConv2d(nn.Conv2d):
    def __init__(
        self,
        mask_type: str,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Union[str, int] = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )

        self.register_buffer("mask", torch.ones_like(self.weight))

        _, _, kH, kW = self.weight.shape

        if mask_type == "A":
            self.mask[:, :, kH // 2, kW // 2 :] = 0
            self.mask[:, :, kH // 2 + 1 :] = 0

        else:
            self.mask[:, :, kH // 2, kW // 2 + 1 :] = 0
            self.mask[:, :, kH // 2 + 1 :] = 0

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # self.weight.data *= self.mask

        return F.conv2d(
            input,
            self.weight * self.mask,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
        )
