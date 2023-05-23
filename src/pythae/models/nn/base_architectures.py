import torch
import torch.nn as nn


class BaseEncoder(nn.Module):
    """This is a base class for Encoders neural networks."""

    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, x):
        r"""This function must be implemented in a child class.
        It takes the input data and returns an instance of
        :class:`~pythae.models.base.base_utils.ModelOutput`.
        If you decide to provide your own encoder network, you must make sure your
        model inherit from this class by setting and then defining your forward function as
        such:

        .. code-block::

            >>> from pythae.models.nn import BaseEncoder
            >>> from pythae.models.base.base_utils import ModelOutput
            ...
            >>> class My_Encoder(BaseEncoder):
            ...
            ...     def __init__(self):
            ...         BaseEncoder.__init__(self)
            ...         # your code
            ...
            ...     def forward(self, x: torch.Tensor):
            ...         # your code
            ...         output = ModelOutput(
            ...             embedding=embedding,
            ...             log_covariance=log_var # for VAE based models
            ...         )
            ...         return output

        Parameters:
            x (torch.Tensor): The input data that must be encoded

        Returns:
            output (~pythae.models.base.base_utils.ModelOutput): The output of the encoder
        """
        raise NotImplementedError()


class BaseDecoder(nn.Module):
    """This is a base class for Decoders neural networks."""

    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, z: torch.Tensor):
        r"""This function must be implemented in a child class.
        It takes the input data and returns an instance of
        :class:`~pythae.models.base.base_utils.ModelOutput`.
        If you decide to provide your own decoder network, you must make sure your
        model inherit from this class by setting and then defining your forward function as
        such:

        .. code-block::

            >>> from pythae.models.nn import BaseDecoder
            >>> from pythae.models.base.base_utils import ModelOutput
            ...
            >>> class My_decoder(BaseDecoder):
            ...
            ...    def __init__(self):
            ...        BaseDecoder.__init__(self)
            ...        # your code
            ...
            ...    def forward(self, z: torch.Tensor):
            ...        # your code
            ...        output = ModelOutput(
            ...             reconstruction=reconstruction
            ...         )
            ...        return output

        Parameters:
            z (torch.Tensor): The latent data that must be decoded

        Returns:
            output (~pythae.models.base.base_utils.ModelOutput): The output of the decoder

        .. note::

            By convention, the reconstruction tensors should be in [0, 1] and of shape
            BATCH x channels x ...

        """
        raise NotImplementedError()


class BaseMetric(nn.Module):
    """This is a base class for Metrics neural networks
    (only applicable for Riemannian based VAE)
    """

    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, x):
        r"""This function must be implemented in a child class.
        It takes the input data and returns an instance of
        :class:`~pythae.models.base.base_utils.ModelOutput`.
        If you decide to provide your own metric network, you must make sure your
        model inherit from this class by setting and then defining your forward function as
        such:

        .. code-block::

            >>> from pythae.models.nn import BaseMetric
            >>> from pythae.models.base.base_utils import ModelOutput
            ...
            >>> class My_Metric(BaseMetric):
            ...
            ...    def __init__(self):
            ...        BaseMetric.__init__(self)
            ...        # your code
            ...
            ...    def forward(self, x: torch.Tensor):
            ...        # your code
            ...        output = ModelOutput(
            ...             L=L # L matrices in the metric of  Riemannian based VAE (see docs)
            ...         )
            ...        return output

        Parameters:
            x (torch.Tensor): The input data that must be encoded

        Returns:
            output (~pythae.models.base.base_utils.ModelOutput): The output of the metric
        """
        raise NotImplementedError()


class BaseDiscriminator(nn.Module):
    """This is a base class for Discriminator neural networks."""

    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, x):
        r"""This function must be implemented in a child class.
        It takes the input data and returns an instance of
        :class:`~pythae.models.base.base_utils.ModelOutput`.
        If you decide to provide your own disctriminator network, you must make sure your
        model inherit from this class by setting and then defining your forward function as
        such:

        .. code-block::

            >>> from pythae.models.nn import BaseDiscriminator
            >>> from pythae.models.base.base_utils import ModelOutput
            ...
            >>> class My_Discriminator(BaseDiscriminator):
            ...
            ...     def __init__(self):
            ...         BaseDiscriminator.__init__(self)
            ...         # your code
            ...
            ...     def forward(self, x: torch.Tensor):
            ...         # your code
            ...         output = ModelOutput(
            ...             adversarial_cost=adversarial_cost
            ...         )
            ...         return output

        Parameters:
            x (torch.Tensor): The input data that must be encoded

        Returns:
            output (~pythae.models.base.base_utils.ModelOutput): The output of the encoder
        """
        raise NotImplementedError()
