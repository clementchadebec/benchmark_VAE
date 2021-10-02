.. _making-your-own-vae:

##################################
Making your own autoencoder model
##################################


By default, the VAE models use Multi Layer Perceptron neural networks
for the encoder and decoder and metric (if applicable) which automatically adapt to the input data shape. The only thing that is needed is to state the data input dimension which equals to ``n_channels x height x width x ...`` in the :class:`ModelConfig`. This important since, if you do not provided any, an error is raised:

.. code-block:: python

    >>> from pyraug.models.base.base_config import BaseModelConfig
    >>> from pyraug.models import BaseVAE
    >>> config = BaseModelConfig()
    >>> BaseVAE(model_config=config)
        Traceback (most recent call last):
          File "<stdin>", line 1, in <module>
          File "/home/clement/Documents/these/implem/pyraug/src/pyraug/models/base/base_vae.py", line 57, in __init__
            raise AttributeError("No input dimension provided !"
        AttributeError: No input dimension provided !'input_dim' parameter of 
            BaseModelConfig instance must be set to 'data_shape' where the shape of the data is [mini_batch x data_shape] . Unable to build encoder automatically

.. note::

    In case you have different size of data, Pyraug will reshape it to the minimum size ``min_n_channels x min_height x min_width x ...``


Hence building a basic network which used the basic provided architectures may be done as follows:

.. code-block:: python

    >>> from pyraug.models.my_model.my_model_config import MyModelConfig
    >>> from pyraug.models.my_model.my_model import MyModelConfig
    >>> config = MyModelConfig(
    ...    input_dim=10 # Setting the data input dimension is needed if you do not use your own autoencoding architecture
    ...    # you parameters goes here
    ... )
    >>> m = MyModel(model_config=config) # Built the model



However, these networks are often not the best suited to generate. Hence, depending on your data, you may want to override the default architecture and use your own networks instead. Doing so is pretty easy! The only thing you have to do is
define you own encoder or decoder ensuring that they 
inherit from the :class:`~pyraug.models.nn.BaseEncoder` or :class:`~pyraug.models.nn.BaseDecoder`.

************************************************
Setting your Encoder
************************************************

To build your on encoder only makes it inherit from :class:`~pyraug.models.nn.BaseEncoder`, define your architecture and code the :class:`forward` method.
Your own Encoder should look as follows:


.. code-block:: python

    >>> from pyraug.models.nn import BaseEncoder

    >>> class MyEncoder(BaseEncoder):
    ...     def __init__(self, args):
    ...         BaseEncoder.__init__(self)
    ...         # your code goes here

    ...     def forward(self, x):
    ...         # your code goes here 
    ...         return mu, log_var

For a complete example, please see tutorial (using_your_architectures.ipynb)

.. warning::
            When building your Encoder, the output order is important. Do not forget to set :math:`\mu` as first argument and the **log** variance then.

************************************************
Setting your decoder
************************************************

Likewise the encoder, to build your on encoder only makes it inherit from :class:`~pyraug.models.nn.BaseDecoder`, define your architecture and code the :class:`forward` method.
Your own Decoder should look as follows:

 .. code-block::

    >>> from pyraug.models.nn import BaseDecoder

    >>> class My_decoder(BaseDecoder):
    ...     def __init__(self):
    ...            BaseDecoder.__init__(self)
    ...            # your code goes here
    
    ...     def forward(self, z):
    ...         # your code goes here
    ...         return mu


For a complete example, please see tutorial (using_your_architectures.ipynb)

.. note::

        By convention, the output tensors :math:`\mu` should be in [0, 1]. Ensure, this is the case when building your net.
