""" Here will be implemented the main Autoencoders architecture

By convention, each implemented model is contained within a folder located in :ref:`pyraug.models`
in which are located 4 modules:

- | *model_config.py*: Contains a :class:`OtherModelConfig` instance inheriting
    from :class:`~pyraug.models.base.BaseModelConfig` where the model configuration is stored and
    a :class:`OtherModelSamplerConfig` instance inheriting from
    :class:`~pyraug.models.base.BaseSamplerConfig` where the configuration of the sampler used
    to generate new samples is defined.
- | *other_model_model.py*: An implementation of the other_model inheriting from
    :class:`~pyraug.models.BaseVAE`.
- | *other_model_sampler.py*: An implementation of the sampler(s) to use to generate new data
    inheriting from :class:`~pyraug.models.base.base_sampler.BaseSampler`.
- *other_model_utils.py*: A module where utils methods are stored.
"""

from .base import BaseAE
from .ae import AE
#from .rhvae.rhvae_model import RHVAE


__all__ = [
    "BaseAE",
    "AE"
]
