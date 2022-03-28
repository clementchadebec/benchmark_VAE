"""Layers useful to build normalizing flows

Code inspired from
- (https://github.com/kamenbliznashki/normalizing_flows)
- (https://github.com/karpathy/pytorch-normalizing-flows)
- (https://github.com/ikostrikov/pytorch-flows)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from pythae.models.base.base_utils import ModelOutput


class MaskedLinear(nn.Linear):
    """Masked Linear Layer inheriting from `~torch.nn.Linear` class and applying a mask to consider 
        only a selection of weight. 
    """
    def __init__(self, in_features, out_features, mask):
        nn.Linear.__init__(self, in_features=in_features, out_features=out_features)

        self.register_buffer('mask', mask)

    def forward(self, x):
        return F.linear(x, self.mask * self.weight, self.bias)


class BatchNorm(nn.Module):
    """A BatchNorm layer used in several flows"""
    def __init__(self, num_features, eps=1e-05, momentum=0.1):
        nn.Module.__init__(self)

        self.eps = eps
        self.momentum = momentum

        self.log_gamma = nn.Parameter(torch.empty(num_features))
        self.beta = nn.Parameter(torch.empty(num_features))

        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        if self.training:
            self.batch_mean = x.mean(0)
            self.batch_var = x.var(0)

            self.running_mean.mul_(1 - self.momentum).add_(self.batch_mean.data * self.momentum)
            self.running_var.mul_(1 - self.momentum).add_(self.batch_var.data *  self.momentum)

            mean = self.batch_mean
            var = self.batch_var
        
        else:
            mean = self.running_mean
            var = self.running_var

        y = (
            (x - mean) / (var + self.eps).sqrt()
        ) * self.log_gamma.exp() + self.beta

        log_abs_det_jac = self.log_gamma - 0.5 * (var + self.eps).log()

        output = ModelOutput(
            out=y,
            log_abs_det_jac = log_abs_det_jac.expand_as(x)
        )

        return output

    def inverse(self, y):
        if self.training:
            mean = self.batch_mean
            var = self.batch_mean

        else:
            mean = self.running_mean
            var = self.running_var

        x = (y - self.beta) * (-self.log_gamma).exp() * (var + self.eps).sqrt() + mean

        log_abs_det_jac = -self.log_gamma + 0.5 * (var + self.eps).log()

        output = ModelOutput(
            out=x,
            log_abs_det_jac = log_abs_det_jac.expand_as(x)
        )

        return output
    
