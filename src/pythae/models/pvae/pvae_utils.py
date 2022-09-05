"""Distributions and manifold taken from
    (https://github.com/emilemathieu/pvae/blob/master/pvae) and 
    (https://github.com/geoopt/geoopt/blob/master/geoopt/manifolds)

"""
import math
from numbers import Number
from typing import Optional, Tuple

import numpy as np
import torch
import torch.distributions as dist
from torch.autograd import Function, grad
from torch.distributions import Normal
from torch.distributions.utils import _standard_normal, broadcast_all
from torch.nn import functional as F

MIN_NORM = 1e-15
BALL_EPS = {torch.float32: 4e-3, torch.float64: 1e-5}


def log_sum_exp_signs(value, signs, dim=0, keepdim=False):
    m, _ = torch.max(value, dim=dim, keepdim=True)
    value0 = value - m
    if keepdim is False:
        m = m.squeeze(dim)
    return m + torch.log(torch.sum(signs * torch.exp(value0), dim=dim, keepdim=keepdim))


def rexpand(A, *dimensions):
    """Expand tensor, adding new dimensions on right."""
    return A.view(A.shape + (1,) * len(dimensions)).expand(A.shape + tuple(dimensions))


def logsinh(x):
    # torch.log(sinh(x))
    return x + torch.log(1 - torch.exp(-2 * x)) - math.log(2)


def tanh(x):  ## OK
    return x.clamp(-15, 15).tanh()


def arsinh(x: torch.Tensor):  ## OK
    return (x + torch.sqrt(1 + x.pow(2))).clamp_min(MIN_NORM).log().to(x.dtype)


def artanh(x: torch.Tensor):  ## OK
    x = x.clamp(-1 + 1e-5, 1 - 1e-5)
    return (torch.log_(1 + x).sub_(torch.log_(1 - x))).mul_(0.5)


def _lambda_x(x, c, keepdim: bool = False, dim: int = -1):  ## OK
    return 2 / (1 - c * x.pow(2).sum(dim=dim, keepdim=keepdim)).clamp_min(MIN_NORM)


def _mobius_add(x, y, c, dim=-1):  ## OK
    x2 = x.pow(2).sum(dim=dim, keepdim=True)
    y2 = y.pow(2).sum(dim=dim, keepdim=True)
    xy = (x * y).sum(dim=dim, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
    return num / denom.clamp_min(MIN_NORM)


def _mobius_scalar_mul(r, x, c, dim: int = -1):  ## OK
    x_norm = x.norm(dim=dim, keepdim=True, p=2).clamp_min(MIN_NORM)
    sqrt_c = c ** 0.5
    res_c = tanh(r * artanh(sqrt_c * x_norm)) * x / (x_norm * sqrt_c)
    return res_c


def _project(x, c, dim: int = -1, eps: float = None):  ## OK
    norm = x.norm(dim=dim, keepdim=True, p=2).clamp_min(MIN_NORM)
    if eps is None:
        eps = BALL_EPS[x.dtype]
    maxnorm = (1 - eps) / (c ** 0.5)
    cond = norm > maxnorm
    projected = x / norm * maxnorm
    return torch.where(cond, projected, x)


def _gyration(u, v, w, c, dim: int = -1):  ## OK
    u2 = u.pow(2).sum(dim=dim, keepdim=True)
    v2 = v.pow(2).sum(dim=dim, keepdim=True)
    uv = (u * v).sum(dim=dim, keepdim=True)
    uw = (u * w).sum(dim=dim, keepdim=True)
    vw = (v * w).sum(dim=dim, keepdim=True)
    c2 = c ** 2
    a = -c2 * uw * v2 + c * vw + 2 * c2 * uv * vw
    b = -c2 * vw * u2 - c * uw
    d = 1 + 2 * c * uv + c2 * u2 * v2
    return w + 2 * (a * u + b * v) / d.clamp_min(MIN_NORM)


class PoincareBall:
    def __init__(self, dim, c=1.0):
        self.c = c
        self.dim = dim

    @property
    def coord_dim(self):
        return int(self.dim)

    @property
    def zero(self):
        return torch.zeros(1, self.dim).to(self.device)

    # def norm(self, x: torch.Tensor, u: torch.Tensor, *, keepdim=False, dim=-1
    # ) -> torch.Tensor: ## OK
    #    return _lambda_x(x, c=self.c, keepdim=keepdim, dim=dim) * u.norm(
    #    dim=dim, keepdim=keepdim, p=2
    # )

    def dist(  ## OK
        self, x: torch.Tensor, y: torch.Tensor, *, keepdim=False, dim=-1
    ) -> torch.Tensor:  ## OK
        sqrt_c = self.c ** 0.5
        dist_c = artanh(
            sqrt_c
            * _mobius_add(-x, y, self.c, dim=dim).norm(dim=dim, p=2, keepdim=keepdim)
        )
        return dist_c * 2 / sqrt_c

    def lambda_x(
        self, x: torch.Tensor, *, dim=-1, keepdim=False
    ) -> torch.Tensor:  ## OK
        return _lambda_x(x, c=self.c, dim=dim, keepdim=keepdim)

    def mobius_add(
        self, x: torch.Tensor, y: torch.Tensor, *, dim=-1, project=True
    ) -> torch.Tensor:  ## OK
        res = _mobius_add(x, y, c=self.c, dim=dim)
        if project:
            return _project(res, c=self.c, dim=dim)
        else:
            return res

    def logmap0(self, x: torch.Tensor, y: torch.Tensor, *, dim=-1) -> torch.Tensor:
        sqrt_c = self.c ** 0.5
        y_norm = y.norm(dim=dim, p=2, keepdim=True).clamp_min(MIN_NORM)
        return y / y_norm / sqrt_c * artanh(sqrt_c * y_norm)

    def logmap(
        self, x: torch.Tensor, y: torch.Tensor, *, dim=-1
    ) -> torch.Tensor:  ## OK
        sub = _mobius_add(-x, y, self.c, dim=dim)
        sub_norm = sub.norm(dim=dim, p=2, keepdim=True).clamp_min(MIN_NORM)
        lam = _lambda_x(x, self.c, keepdim=True, dim=dim)
        sqrt_c = self.c ** 0.5
        return 2 / sqrt_c / lam * artanh(sqrt_c * sub_norm) * sub / sub_norm

    def transp0(self, y: torch.Tensor, v: torch.Tensor, *, dim=-1) -> torch.Tensor:
        return v * (1 - self.c * y.pow(2).sum(dim=dim, keepdim=True)).clamp_min(
            MIN_NORM
        )

    def transp(
        self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor, *, dim=-1
    ):  ## OK
        return (
            _gyration(y, -x, v, self.c, dim=dim)
            * _lambda_x(x, self.c, keepdim=True, dim=dim)
            / _lambda_x(y, self.c, keepdim=True, dim=dim)
        )

    def logdetexp(self, x, y, is_vector=False, keepdim=False):  ## OK
        d = (
            self.norm(x, y, keepdim=keepdim)
            if is_vector
            else self.dist(x, y, keepdim=keepdim)
        )
        return (self.dim - 1) * (
            torch.sinh(math.sqrt(self.c) * d) / math.sqrt(self.c) / d
        ).log()

    def expmap0(self, u, dim: int = -1):
        sqrt_c = self.c ** 0.5
        u_norm = u.norm(dim=dim, p=2, keepdim=True).clamp_min(MIN_NORM)
        gamma_1 = tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
        return gamma_1

    def expmap(self, x, u, dim: int = -1):
        sqrt_c = self.c ** 0.5
        u_norm = u.norm(dim=dim, p=2, keepdim=True).clamp_min(MIN_NORM)
        second_term = (
            tanh(sqrt_c / 2 * _lambda_x(x, self.c, keepdim=True, dim=dim) * u_norm)
            * u
            / (sqrt_c * u_norm)
        )
        gamma_1 = _mobius_add(x, second_term, self.c, dim=dim)
        return gamma_1

    def expmap_polar(self, x, u, r, dim: int = -1):  ## OK
        sqrt_c = self.c ** 0.5
        u_norm = u.norm(dim=dim, p=2, keepdim=True).clamp_min(MIN_NORM)
        second_term = (
            tanh(torch.tensor([sqrt_c]).to(x.device) / 2 * r) * u / (sqrt_c * u_norm)
        )
        gamma_1 = self.mobius_add(x, second_term, dim=dim)
        return gamma_1

    def geodesic(self, t, x, y, dim: int = -1):  ## OK
        v = _mobius_add(-x, y, self.c, dim=dim)
        tv = _mobius_scalar_mul(t, v, self.c, dim=dim)
        gamma_t = _mobius_add(x, tv, self.c, dim=dim)
        return gamma_t

    def normdist2plane(
        self,
        x,
        a,
        p,
        keepdim: bool = False,
        signed: bool = False,
        dim: int = -1,
        norm: bool = False,
    ):
        c = self.c
        sqrt_c = c ** 0.5
        diff = self.mobius_add(-p, x, dim=dim)
        diff_norm2 = diff.pow(2).sum(dim=dim, keepdim=keepdim).clamp_min(MIN_NORM)
        sc_diff_a = (diff * a).sum(dim=dim, keepdim=keepdim)
        if not signed:
            sc_diff_a = sc_diff_a.abs()
        a_norm = a.norm(dim=dim, keepdim=keepdim, p=2).clamp_min(MIN_NORM)
        num = 2 * sqrt_c * sc_diff_a
        denom = (1 - c * diff_norm2) * a_norm
        res = arsinh(num / denom.clamp_min(MIN_NORM)) / sqrt_c
        if norm:
            res = res * a_norm  # * self.lambda_x(a, dim=dim, keepdim=keepdim)
        return res

    def _check_point_on_manifold(self, x, *, atol=1e-5, rtol=1e-5):
        px = _project(x, c=self.c)
        ok = torch.allclose(x, px, atol=atol, rtol=rtol)
        if not ok:
            reason = "'x' norm lies out of the bounds [-1/sqrt(c)+eps, 1/sqrt(c)-eps]"
        else:
            reason = None
        return ok, reason

    def _check_vector_on_tangent(
        self, x: torch.Tensor, u: torch.Tensor, *, atol=1e-5, rtol=1e-5, dim=-1
    ) -> Tuple[bool, Optional[str]]:
        return True, None


class WrappedNormal(dist.Distribution):  ## OK
    """Wrapped Normal distribution"""

    arg_constraints = {"loc": dist.constraints.real, "scale": dist.constraints.positive}
    support = dist.constraints.real
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def scale(self):
        return F.softplus(self._scale) if self.softplus else self._scale

    def __init__(self, loc, scale, manifold, validate_args=None, softplus=False):
        self.dtype = loc.dtype
        self.softplus = softplus
        self.loc, self._scale = broadcast_all(loc, scale)
        self.manifold = manifold
        self.manifold._check_point_on_manifold(self.loc)
        self.device = loc.device
        if isinstance(loc, Number) and isinstance(scale, Number):
            batch_shape, event_shape = torch.Size(), torch.Size()
        else:
            batch_shape = self.loc.shape[:-1]
            event_shape = torch.Size([self.manifold.dim])
        super(WrappedNormal, self).__init__(
            batch_shape, event_shape, validate_args=validate_args
        )

    def sample(self, shape=torch.Size()):  ## OK
        with torch.no_grad():
            return self.rsample(shape)

    def rsample(self, sample_shape=torch.Size()):  ## OK
        shape = self._extended_shape(sample_shape)
        v = self.scale * _standard_normal(
            shape, dtype=self.loc.dtype, device=self.loc.device
        )
        self.manifold._check_vector_on_tangent(
            torch.zeros(1, self.manifold.dim).to(v.device), v
        )
        v = v / self.manifold.lambda_x(
            torch.zeros(1, self.manifold.dim).to(v.device), keepdim=True
        )
        u = self.manifold.transp(
            torch.zeros(1, self.manifold.dim).to(v.device), self.loc, v
        )
        z = self.manifold.expmap(self.loc, u)
        return z

    def log_prob(self, x):  ## OK
        shape = x.shape
        loc = self.loc.unsqueeze(0).expand(
            x.shape[0], *self.batch_shape, self.manifold.coord_dim
        )
        if len(shape) < len(loc.shape):
            x = x.unsqueeze(1)
        v = self.manifold.logmap(loc, x)
        v = self.manifold.transp(loc, torch.zeros(1, self.manifold.dim).to(v.device), v)
        u = v * self.manifold.lambda_x(
            torch.zeros(1, self.manifold.dim).to(v.device), keepdim=True
        )
        norm_pdf = (
            Normal(torch.zeros_like(self.scale), self.scale)
            .log_prob(u)
            .sum(-1, keepdim=True)
        )
        logdetexp = self.manifold.logdetexp(loc, x, keepdim=True)
        result = norm_pdf - logdetexp
        return result


infty = torch.tensor(float("Inf"))


def diff(x):
    return x[:, 1:] - x[:, :-1]


class ARS:
    """
    This class implements the Adaptive Rejection Sampling technique of Gilks and Wild '92.
    Where possible, naming convention has been borrowed from this paper.
    The PDF must be log-concave.
    Currently does not exploit lower hull described in paper- which is fine for drawing
    only small amount of samples at a time.
    """

    def __init__(
        self,
        logpdf,
        grad_logpdf,
        device,
        xi,
        lb=-infty,
        ub=infty,
        use_lower=False,
        ns=50,
        **fargs,
    ):
        """
        initialize the upper (and if needed lower) hulls with the specified params

        Parameters
        ==========
        f: function that computes log(f(u,...)), for given u, where f(u) is proportional to the
           density we want to sample from
        fprima:  d/du log(f(u,...))
        xi: ordered vector of starting points in wich log(f(u,...) is defined
            to initialize the hulls
        use_lower: True means the lower sqeezing will be used; which is more efficient
                   for drawing large numbers of samples


        lb: lower bound of the domain
        ub: upper bound of the domain
        ns: maximum number of points defining the hulls
        fargs: arguments for f and fprima
        """
        self.device = device

        self.lb = lb
        self.ub = ub

        self.logpdf = logpdf
        self.grad_logpdf = grad_logpdf
        self.fargs = fargs

        # set limit on how many points to maintain on hull
        self.ns = ns
        self.xi = xi.to(
            self.device
        )  # initialize x, the vector of absicassae at which the function h has been evaluated
        self.B, self.K = self.xi.size()  # hull size
        self.h = torch.zeros(self.B, ns).to(self.device)
        self.hprime = torch.zeros(self.B, ns).to(self.device)
        self.x = torch.zeros(self.B, ns).to(self.device)
        self.h[:, : self.K] = self.logpdf(self.xi, **self.fargs)
        self.hprime[:, : self.K] = self.grad_logpdf(self.xi, **self.fargs)
        self.x[:, : self.K] = self.xi
        # Avoid under/overflow errors. the envelope and pdf are only
        # proportional to the true pdf, so can choose any constant of proportionality.
        self.offset = self.h.max(-1)[0].view(-1, 1)
        self.h = self.h - self.offset

        # Derivative at first point in xi must be > 0
        # Derivative at last point in xi must be < 0
        if not (self.hprime[:, 0] > 0).all():
            raise IOError("initial anchor points must span mode of PDF (left)")
        if not (self.hprime[:, self.K - 1] < 0).all():
            raise IOError("initial anchor points must span mode of PDF (right)")
        self.insert()

    def sample(self, shape=torch.Size()):
        """
        Draw N samples and update upper and lower hulls accordingly
        """
        shape = shape if isinstance(shape, torch.Size) else torch.Size([shape])
        samples = torch.ones(self.B, *shape).to(self.device)
        bool_mask = (torch.ones(self.B, *shape) == 1).to(self.device)
        count = 0
        while bool_mask.sum() != 0:
            count += 1
            xt, i = self.sampleUpper(shape)
            ht = self.logpdf(xt, **self.fargs)
            # hprimet = self.grad_logpdf(xt, **self.fargs)
            ht = ht - self.offset
            ut = self.h.gather(1, i) + (xt - self.x.gather(1, i)) * self.hprime.gather(
                1, i
            )

            # Accept sample?
            u = torch.rand(shape).to(self.device)
            accept = u < torch.exp(ht - ut)
            reject = ~accept
            samples[bool_mask * accept] = xt[bool_mask * accept]
            bool_mask[bool_mask * accept] = reject[bool_mask * accept]
            # Update hull with new function evaluations
            # if self.K < self.ns:
            #     nb_insert = self.ns - self.K
            #     self.insert(nb_insert, xt[:, :nb_insert], ht[:, :nb_insert], hprimet[:, :nb_insert])

        return samples.t().unsqueeze(-1)

    def insert(self, nbnew=0, xnew=None, hnew=None, hprimenew=None):
        """
        Update hulls with new point(s) if none given, just recalculate hull from existing x,h,hprime
        #"""
        self.z = torch.zeros(self.B, self.K + 1).to(self.device)
        self.z[:, 0] = self.lb
        self.z[:, self.K] = self.ub
        self.z[:, 1 : self.K] = (
            diff(self.h[:, : self.K])
            - diff(self.x[:, : self.K] * self.hprime[:, : self.K])
        ) / -diff(self.hprime[:, : self.K])
        idx = [0] + list(range(self.K))
        self.u = self.h[:, idx] + self.hprime[:, idx] * (self.z - self.x[:, idx])

        self.s = diff(torch.exp(self.u)) / self.hprime[:, : self.K]
        self.s[self.hprime[:, : self.K] == 0.0] = 0.0  # should be 0 when gradient is 0
        self.cs = torch.cat(
            (torch.zeros(self.B, 1).to(self.device), torch.cumsum(self.s, dim=-1)),
            dim=-1,
        )
        self.cu = self.cs[:, -1]

    def sampleUpper(self, shape=torch.Size()):
        """
        Return a single value randomly sampled from the upper hull and index of segment
        """

        u = torch.rand(self.B, *shape).to(self.device)
        i = (self.cs / self.cu.unsqueeze(-1)).unsqueeze(-1) <= u.unsqueeze(1).expand(
            *self.cs.shape, *shape
        )
        idx = i.sum(1) - 1

        xt = self.x.gather(1, idx) + (
            -self.h.gather(1, idx)
            + torch.log(
                self.hprime.gather(1, idx)
                * (self.cu.unsqueeze(-1) * u - self.cs.gather(1, idx))
                + torch.exp(self.u.gather(1, idx))
            )
        ) / self.hprime.gather(1, idx)

        return xt, idx


def cdf_r(value, scale, c, dim):
    value = value.double()
    scale = scale.double()
    c = np.double(c)

    if dim == 2:
        return (
            1
            / torch.erf(math.sqrt(c) * scale / math.sqrt(2))
            * 0.5
            * (
                2 * torch.erf(math.sqrt(c) * scale / math.sqrt(2))
                + torch.erf(
                    (value - math.sqrt(c) * scale.pow(2)) / math.sqrt(2) / scale
                )
                - torch.erf(
                    (math.sqrt(c) * scale.pow(2) + value) / math.sqrt(2) / scale
                )
            )
        )
    else:
        device = value.device

        k_float = rexpand(torch.arange(dim), *value.size()).double().to(device)
        dim = torch.tensor(dim).to(device).double()

        s1 = (
            torch.lgamma(dim)
            - torch.lgamma(k_float + 1)
            - torch.lgamma(dim - k_float)
            + (dim - 1 - 2 * k_float).pow(2) * c * scale.pow(2) / 2
            + torch.log(
                torch.erf(
                    (value - (dim - 1 - 2 * k_float) * math.sqrt(c) * scale.pow(2))
                    / scale
                    / math.sqrt(2)
                )
                + torch.erf(
                    (dim - 1 - 2 * k_float) * math.sqrt(c) * scale / math.sqrt(2)
                )
            )
        )
        s2 = (
            torch.lgamma(dim)
            - torch.lgamma(k_float + 1)
            - torch.lgamma(dim - k_float)
            + (dim - 1 - 2 * k_float).pow(2) * c * scale.pow(2) / 2
            + torch.log1p(
                torch.erf((dim - 1 - 2 * k_float) * math.sqrt(c) * scale / math.sqrt(2))
            )
        )

        signs = (
            torch.tensor([1.0, -1.0])
            .double()
            .to(device)
            .repeat(((int(dim) + 1) // 2) * 2)[: int(dim)]
        )
        signs = rexpand(signs, *value.size())

        S1 = log_sum_exp_signs(s1, signs, dim=0)
        S2 = log_sum_exp_signs(s2, signs, dim=0)

        output = torch.exp(S1 - S2)
        zero_value_idx = value == 0.0
        output[zero_value_idx] = 0.0
        return output.float()


def grad_cdf_value_scale(value, scale, c, dim):
    device = value.device

    dim = torch.tensor(int(dim)).to(device).double()

    signs = (
        torch.tensor([1.0, -1.0])
        .double()
        .to(device)
        .repeat(((int(dim) + 1) // 2) * 2)[: int(dim)]
    )
    signs = rexpand(signs, *value.size())
    k_float = rexpand(torch.arange(dim), *value.size()).double().to(device)

    log_arg1 = (
        (dim - 1 - 2 * k_float).pow(2)
        * c
        * scale
        * (
            torch.erf(
                (value - (dim - 1 - 2 * k_float) * math.sqrt(c) * scale.pow(2))
                / scale
                / math.sqrt(2)
            )
            + torch.erf((dim - 1 - 2 * k_float) * math.sqrt(c) * scale / math.sqrt(2))
        )
    )

    log_arg2 = math.sqrt(2 / math.pi) * (
        (dim - 1 - 2 * k_float)
        * math.sqrt(c)
        * torch.exp(-(dim - 1 - 2 * k_float).pow(2) * c * scale.pow(2) / 2)
        - (
            (value / scale.pow(2) + (dim - 1 - 2 * k_float) * math.sqrt(c))
            * torch.exp(
                -(value - (dim - 1 - 2 * k_float) * math.sqrt(c) * scale.pow(2)).pow(2)
                / (2 * scale.pow(2))
            )
        )
    )

    log_arg = log_arg1 + log_arg2
    sign_log_arg = torch.sign(log_arg)

    s = (
        torch.lgamma(dim)
        - torch.lgamma(k_float + 1)
        - torch.lgamma(dim - k_float)
        + (dim - 1 - 2 * k_float).pow(2) * c * scale.pow(2) / 2
        + torch.log(sign_log_arg * log_arg)
    )

    log_grad_sum_sigma = log_sum_exp_signs(s, signs * sign_log_arg, dim=0)
    grad_sum_sigma = torch.sum(signs * sign_log_arg * torch.exp(s), dim=0)

    s1 = (
        torch.lgamma(dim)
        - torch.lgamma(k_float + 1)
        - torch.lgamma(dim - k_float)
        + (dim - 1 - 2 * k_float).pow(2) * c * scale.pow(2) / 2
        + torch.log(
            torch.erf(
                (value - (dim - 1 - 2 * k_float) * math.sqrt(c) * scale.pow(2))
                / scale
                / math.sqrt(2)
            )
            + torch.erf((dim - 1 - 2 * k_float) * math.sqrt(c) * scale / math.sqrt(2))
        )
    )

    S1 = log_sum_exp_signs(s1, signs, dim=0)
    grad_log_cdf_scale = grad_sum_sigma / S1.exp()
    log_unormalised_prob = (
        -value.pow(2) / (2 * scale.pow(2))
        + (dim - 1) * logsinh(math.sqrt(c) * value)
        - (dim - 1) / 2 * math.log(c)
    )

    with torch.autograd.enable_grad():
        scale = scale.float()
        logZ = _log_normalizer_closed_grad.apply(scale, c, dim)
        grad_logZ_scale = grad(logZ, scale, grad_outputs=torch.ones_like(scale))

    grad_log_cdf_scale = -grad_logZ_scale[0] + 1 / scale + grad_log_cdf_scale.float()
    cdf = (
        cdf_r(value.double(), scale.double(), np.double(c), int(dim)).float().squeeze(0)
    )
    grad_scale = cdf * grad_log_cdf_scale

    grad_value = (log_unormalised_prob.float() - logZ).exp()
    return grad_value, grad_scale


class _log_normalizer_closed_grad(Function):
    @staticmethod
    def forward(ctx, scale, c, dim):
        scale = scale.double()
        c = np.double(c)
        ctx.scale = scale.clone().detach()
        ctx.c = torch.tensor([c]).to(scale.device)
        ctx.dim = dim

        device = scale.device
        output = (
            0.5 * (math.log(math.pi) - math.log(2))
            + scale.log()
            - (int(dim) - 1) * (math.log(c) / 2 + math.log(2))
        )
        dim = torch.tensor(int(dim)).to(device).double()

        k_float = rexpand(torch.arange(int(dim)), *scale.size()).double().to(device)
        s = (
            torch.lgamma(dim)
            - torch.lgamma(k_float + 1)
            - torch.lgamma(dim - k_float)
            + (dim - 1 - 2 * k_float).pow(2) * c * scale.pow(2) / 2
            + torch.log1p(
                torch.erf((dim - 1 - 2 * k_float) * math.sqrt(c) * scale / math.sqrt(2))
            )
        )
        signs = (
            torch.tensor([1.0, -1.0])
            .double()
            .to(device)
            .repeat(((int(dim) + 1) // 2) * 2)[: int(dim)]
        )
        signs = rexpand(signs, *scale.size())
        ctx.log_sum_term = log_sum_exp_signs(s, signs, dim=0)
        output = output + ctx.log_sum_term

        return output.float()

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()

        device = grad_input.device
        scale = ctx.scale
        c = ctx.c
        dim = torch.tensor(int(ctx.dim)).to(device).double()

        k_float = rexpand(torch.arange(int(dim)), *scale.size()).double().to(device)
        signs = (
            torch.tensor([1.0, -1.0])
            .double()
            .to(device)
            .repeat(((int(dim) + 1) // 2) * 2)[: int(dim)]
        )
        signs = rexpand(signs, *scale.size())

        log_arg = (dim - 1 - 2 * k_float).pow(2) * c * scale * (
            1 + torch.erf((dim - 1 - 2 * k_float) * math.sqrt(c) * scale / math.sqrt(2))
        ) + torch.exp(
            -(dim - 1 - 2 * k_float).pow(2) * c * scale.pow(2) / 2
        ) * 2 / math.sqrt(
            math.pi
        ) * (
            dim - 1 - 2 * k_float
        ) * math.sqrt(
            c
        ) / math.sqrt(
            2
        )
        log_arg_signs = torch.sign(log_arg)
        s = (
            torch.lgamma(dim)
            - torch.lgamma(k_float + 1)
            - torch.lgamma(dim - k_float)
            + (dim - 1 - 2 * k_float).pow(2) * c * scale.pow(2) / 2
            + torch.log(log_arg_signs * log_arg)
        )
        log_grad_sum_sigma = log_sum_exp_signs(s, log_arg_signs * signs, dim=0)

        grad_scale = torch.exp(log_grad_sum_sigma - ctx.log_sum_term)
        grad_scale = 1 / ctx.scale + grad_scale

        grad_scale = (
            (grad_input * grad_scale.float()).view(-1, *grad_input.shape).sum(0)
        )
        return (grad_scale, None, None)


class impl_rsample(Function):
    @staticmethod
    def forward(ctx, value, scale, c, dim):
        ctx.scale = scale.clone().detach().double().requires_grad_(True)
        ctx.value = value.clone().detach().double().requires_grad_(True)
        ctx.c = torch.tensor([c]).to(scale.device).double().requires_grad_(True)
        ctx.dim = dim
        return value

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        grad_cdf_value, grad_cdf_scale = grad_cdf_value_scale(
            ctx.value, ctx.scale, ctx.c, ctx.dim
        )
        assert not torch.isnan(grad_cdf_value).any()
        assert not torch.isnan(grad_cdf_scale).any()
        grad_value_scale = -(grad_cdf_value).pow(-1) * grad_cdf_scale.expand(
            grad_input.shape
        )
        grad_scale = (
            (grad_input * grad_value_scale).view(-1, *grad_cdf_scale.shape).sum(0)
        )
        # grad_value_c = -(grad_cdf_value).pow(-1) * grad_cdf_c.expand(grad_input.shape)
        # grad_c = (grad_input * grad_value_c).view(-1, *grad_cdf_c.shape).sum(0)
        return (None, grad_scale, None, None)


class HyperbolicRadius(dist.Distribution):
    support = dist.constraints.positive
    has_rsample = True

    def __init__(self, dim, c, scale, ars=True, validate_args=None):
        self.dim = dim
        self.c = c
        self.scale = scale
        self.device = scale.device
        self.ars = ars
        if isinstance(scale, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.scale.size()
        self.log_normalizer = self._log_normalizer()
        if (
            torch.isnan(self.log_normalizer).any()
            or torch.isinf(self.log_normalizer).any()
        ):
            print(
                "nan or inf in log_normalizer",
                torch.cat((self.log_normalizer, self.scale), dim=1),
            )
            raise
        super(HyperbolicRadius, self).__init__(batch_shape)

    def rsample(self, sample_shape=torch.Size()):
        value = self.sample(sample_shape)
        return impl_rsample.apply(value, self.scale, self.c, self.dim)

    def sample(self, sample_shape=torch.Size()):
        if sample_shape == torch.Size():
            sample_shape = torch.Size([1])
        with torch.no_grad():
            mean = self.mean
            stddev = self.stddev
            if torch.isnan(stddev).any():
                stddev[torch.isnan(stddev)] = self.scale[torch.isnan(stddev)]
            if torch.isnan(mean).any():
                mean[torch.isnan(mean)] = (
                    (self.dim - 1) * self.scale.pow(2) * math.sqrt(self.c)
                )[torch.isnan(mean)]
            steps = torch.linspace(0.1, 3, 10).to(self.device)
            steps = torch.cat((-steps.flip(0), steps))
            xi = [mean + s * torch.min(stddev, 0.95 * mean / 3) for s in steps]
            xi = torch.cat(xi, dim=1)
            ars = ARS(
                self.log_prob, self.grad_log_prob, self.device, xi=xi, ns=20, lb=0
            )
            value = ars.sample(sample_shape)
        return value

    def log_prob(self, value):
        res = (
            -value.pow(2) / (2 * self.scale.pow(2))
            + (self.dim - 1) * logsinh(math.sqrt(self.c) * value)
            - (self.dim - 1) / 2 * math.log(self.c)
            - self.log_normalizer
        )  # .expand(value.shape)
        assert not torch.isnan(res).any()
        return res

    def grad_log_prob(self, value):
        res = -value / self.scale.pow(2) + (self.dim - 1) * math.sqrt(
            self.c
        ) * torch.cosh(math.sqrt(self.c) * value) / torch.sinh(
            math.sqrt(self.c) * value
        )
        return res

    def cdf(self, value):
        return cdf_r(value, self.scale, self.c, self.dim)

    @property
    def mean(self):
        c = np.double(self.c)
        scale = self.scale.double()
        dim = torch.tensor(int(self.dim)).double().to(self.device)
        signs = (
            torch.tensor([1.0, -1.0])
            .double()
            .to(self.device)
            .repeat(((self.dim + 1) // 2) * 2)[: self.dim]
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(self.dim, *self.scale.size())
        )

        k_float = (
            rexpand(torch.arange(self.dim), *self.scale.size()).double().to(self.device)
        )
        s2 = (
            torch.lgamma(dim)
            - torch.lgamma(k_float + 1)
            - torch.lgamma(dim - k_float)
            + (dim - 1 - 2 * k_float).pow(2) * c * scale.pow(2) / 2
            + torch.log1p(
                torch.erf((dim - 1 - 2 * k_float) * math.sqrt(c) * scale / math.sqrt(2))
            )
        )
        S2 = log_sum_exp_signs(s2, signs, dim=0)

        log_arg = (dim - 1 - 2 * k_float) * math.sqrt(c) * scale.pow(2) * (
            1 + torch.erf((dim - 1 - 2 * k_float) * math.sqrt(c) * scale / math.sqrt(2))
        ) + torch.exp(
            -(dim - 1 - 2 * k_float).pow(2) * c * scale.pow(2) / 2
        ) * scale * math.sqrt(
            2 / math.pi
        )
        log_arg_signs = torch.sign(log_arg)
        s1 = (
            torch.lgamma(dim)
            - torch.lgamma(k_float + 1)
            - torch.lgamma(dim - k_float)
            + (dim - 1 - 2 * k_float).pow(2) * c * scale.pow(2) / 2
            + torch.log(log_arg_signs * log_arg)
        )
        S1 = log_sum_exp_signs(s1, signs * log_arg_signs, dim=0)

        output = torch.exp(S1 - S2)
        return output.float()

    @property
    def variance(self):
        c = np.double(self.c)
        scale = self.scale.double()
        dim = torch.tensor(int(self.dim)).double().to(self.device)
        signs = (
            torch.tensor([1.0, -1.0])
            .double()
            .to(self.device)
            .repeat(((int(dim) + 1) // 2) * 2)[: int(dim)]
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(int(dim), *self.scale.size())
        )

        k_float = (
            rexpand(torch.arange(self.dim), *self.scale.size()).double().to(self.device)
        )
        s2 = (
            torch.lgamma(dim)
            - torch.lgamma(k_float + 1)
            - torch.lgamma(dim - k_float)
            + (dim - 1 - 2 * k_float).pow(2) * c * scale.pow(2) / 2
            + torch.log1p(
                torch.erf((dim - 1 - 2 * k_float) * math.sqrt(c) * scale / math.sqrt(2))
            )
        )
        S2 = log_sum_exp_signs(s2, signs, dim=0)

        log_arg = (1 + (dim - 1 - 2 * k_float).pow(2) * c * scale.pow(2)) * (
            1 + torch.erf((dim - 1 - 2 * k_float) * math.sqrt(c) * scale / math.sqrt(2))
        ) + (dim - 1 - 2 * k_float) * math.sqrt(c) * torch.exp(
            -(dim - 1 - 2 * k_float).pow(2) * c * scale.pow(2) / 2
        ) * scale * math.sqrt(
            2 / math.pi
        )
        log_arg_signs = torch.sign(log_arg)
        s1 = (
            torch.lgamma(dim)
            - torch.lgamma(k_float + 1)
            - torch.lgamma(dim - k_float)
            + (dim - 1 - 2 * k_float).pow(2) * c * scale.pow(2) / 2
            + 2 * scale.log()
            + torch.log(log_arg_signs * log_arg)
        )
        S1 = log_sum_exp_signs(s1, signs * log_arg_signs, dim=0)

        output = torch.exp(S1 - S2)
        output = output.float() - self.mean.pow(2)
        return output

    @property
    def stddev(self):
        return self.variance.sqrt()

    def _log_normalizer(self):
        return _log_normalizer_closed_grad.apply(self.scale, self.c, self.dim)


class HypersphericalUniform(dist.Distribution):
    """Taken from
    https://github.com/nicola-decao/s-vae-pytorch/blob/master/hyperspherical_vae/distributions/von_mises_fisher.py
    """

    support = dist.constraints.real
    has_rsample = False
    _mean_carrier_measure = 0

    @property
    def dim(self):
        return self._dim

    def __init__(self, dim, device="cpu", validate_args=None):
        super(HypersphericalUniform, self).__init__(
            torch.Size([dim]), validate_args=validate_args
        )
        self._dim = dim
        self._device = device

    def sample(self, shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(shape)

    def rsample(self, sample_shape=torch.Size()):
        shape = torch.Size([*sample_shape, self._dim + 1])
        output = _standard_normal(shape, dtype=torch.float, device=self._device)

        return output / output.norm(dim=-1, keepdim=True)

    def entropy(self):
        return self.__log_surface_area()

    def log_prob(self, x):
        return -torch.ones(x.shape[:-1]).to(self._device) * self._log_normalizer()

    def _log_normalizer(self):
        return self._log_surface_area().to(self._device)

    def _log_surface_area(self):
        return (
            math.log(2)
            + ((self._dim + 1) / 2) * math.log(math.pi)
            - torch.lgamma(torch.Tensor([(self._dim + 1) / 2]))
        )


class RiemannianNormal(dist.Distribution):  ## OK
    # arg_constraints = {'loc': dist.constraints.interval(-1, 1), 'scale': dist.constraints.positive}
    support = dist.constraints.interval(-1, 1)
    has_rsample = True

    @property
    def mean(self):
        return self.loc

    def __init__(self, loc, scale, manifold, validate_args=None):
        assert not (torch.isnan(loc).any() or torch.isnan(scale).any())
        self.manifold = manifold
        self.loc = loc
        self.manifold._check_point_on_manifold(self.loc)
        self.scale = scale.clamp(min=0.1, max=7.0)
        self.radius = HyperbolicRadius(manifold.dim, manifold.c, self.scale)
        self.direction = HypersphericalUniform(manifold.dim - 1, device=loc.device)
        if isinstance(loc, Number) and isinstance(scale, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super(RiemannianNormal, self).__init__(batch_shape, validate_args=validate_args)

    def sample(self, shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(shape)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        alpha = self.direction.sample(torch.Size([*shape[:-1]]))
        radius = self.radius.rsample(sample_shape)
        # u = radius * alpha / self.manifold.lambda_x(self.loc, keepdim=True)
        # res = self.manifold.expmap(self.loc, u)
        res = self.manifold.expmap_polar(self.loc, alpha, radius)
        return res

    def log_prob(self, value):  ## OK
        loc = self.loc.expand(value.shape)
        radius_sq = self.manifold.dist(loc, value, keepdim=True).pow(2)
        res = (
            -radius_sq / 2 / self.scale.pow(2)
            - self.direction._log_normalizer()
            - self.radius.log_normalizer
        )
        return res
