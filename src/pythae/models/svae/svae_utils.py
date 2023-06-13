import scipy.special
import torch

# Using exp(-x)I_v(k) for numerical stability (vanishes in ratios)


class ModifiedBesselFn(torch.autograd.Function):
    @staticmethod
    def forward(self, nu, inp):
        self._nu = nu
        self.save_for_backward(inp)
        inp_cpu = inp.data.cpu().numpy()
        return torch.from_numpy(scipy.special.ive(nu, inp_cpu)).to(inp.device)

    @staticmethod
    def backward(self, grad_out):
        inp = self.saved_tensors[-1]
        nu = self._nu
        return (None, grad_out * (ive(nu - 1, inp) - ive(nu, inp) * (nu + inp) / inp))


ive = ModifiedBesselFn.apply
