import torch.nn as nn

from ..base.base_utils import ModelOuput

class BaseEncoder(nn.Module):
    """This is a base class for Encoders neural networks.
    """

    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, x):
        r"""This function must be implemented in a child class.
        It takes the input data and returns an instance of 
         :class:`~pythae.models.base.base_utils.ModelOutput in that order.
        If you decide to provide your own encoder network, you must make your
        model inherit from this class by setting and the define your forward function as
        such:

        .. code-block::

            class My_Encoder(BaseEncoder):

                def __init__(self):
                    BaseEncoder.__init__(self)
                    # your code

                def forward(self, x):
                    # your code
                    return output

        Parameters:
            x (torch.Tensor): The input data that must be encoded

        Returns:
            output (~pythae.models.base.base_utils.ModelOutput): The output of the encoder
        """
        raise NotImplementedError()


class BaseDecoder(nn.Module):
    """This is a base class for Decoders neural networks.
    """

    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, x):
        r"""This function must be implemented in a child class.
        It takes the latent data and returns (mu). If you decide to provide
        your own decoder network  you must make your
        model inherit from this class by setting and the define your forward function as
        such:

        .. code-block::

            class My_decoder(BaseDecoder):

                def __init__(self):
                    BaseDecoder.__init__(self)
                    # your code

                def forward(self, z):
                    # your code
                    return output

        Parameters:
            z (torch.Tensor): The latent data that must be decoded

        Returns:
            output (~pythae.models.base.base_utils.ModelOutput): The output of the encoder
           
        .. note::

            By convention, the output tensors :math:`\mu` should be in [0, 1]

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
        It takes the input data and returns (L_psi).
        If you decide to provide your own metric network, you must make your
        model inherit from this class by setting and the define your forward function as
        such:

        .. code-block::

            class My_Metric(BaseMetric):

                def __init__(self):
                    BaseMetric.__init__(self)
                    # your code

                def forward(self, x):
                    # your code
                    return L

        Parameters:
            x (torch.Tensor): The input data that must be encoded

        Returns:
            output (~pythae.models.base.base_utils.ModelOutput): The output of the metric
        """
        raise NotImplementedError()
