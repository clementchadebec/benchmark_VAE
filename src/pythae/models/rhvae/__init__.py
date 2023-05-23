r"""This is an implementation of the Riemannian Hamiltonian VAE model proposed in
(https://arxiv.org/abs/2105.00026). This model provides a way to
learn the Riemannian latent structure of a given set of data set through a parametrized
Riemannian metric having the following shape:
:math:`\mathbf{G}^{-1}(z) = \sum \limits _{i=1}^N L_{\psi_i} L_{\psi_i}^{\top} \exp
\Big(-\frac{\lVert z - c_i \rVert_2^2}{T^2} \Big) + \lambda I_d`

It is particularly well suited for High
Dimensional data combined with low sample number and proved relevant for Data Augmentation as
proved in (https://arxiv.org/abs/2105.00026).

Available samplers
-------------------

.. autosummary::
    ~pythae.samplers.RHVAESampler
    ~pythae.samplers.NormalSampler
    ~pythae.samplers.GaussianMixtureSampler
    ~pythae.samplers.TwoStageVAESampler
    ~pythae.samplers.MAFSampler
    ~pythae.samplers.IAFSampler
    :nosignatures:
"""

from .rhvae_config import RHVAEConfig
from .rhvae_model import RHVAE

__all__ = ["RHVAE", "RHVAEConfig"]
