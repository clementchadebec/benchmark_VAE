
.. _setting your config:

##################################
Setting up your own configurations
##################################



The augmentation methods relies on default parameters for the model, training and generation.
Depending on your data these parameters should be modified. 


************************************************
Link between ``.json`` files and ``dataclasses`` 
************************************************


In pyraug, the configurations of the models, trainers and samplers are stored and used as :class:`dataclasses.dataclass` and all inherit from the :class:`~pyraug.config.BaseConfig`. Hence, any configuration class has a classmethod :class:`~pyraug.config.BaseConfig.from_json_file` coming from :class:`~pyraug.config.BaseConfig` allowing to directly load config from ``.json`` files into ``dataclasses``.


.. _loading from json:

Loading a config from a ``.json``
=================================================

Say that you want to load a training configuration that is stored in a ``training_config.json`` file. To convert it in a :class:`~pyraug.trainers.training_config.TrainingConfig` run the following

.. code-block::

    >>> from pyraug.trainers.training_config import TrainingConfig
    >>> config = TrainingConfig.from_json_file(
    ...     'scripts/configs/training_config.json')
    >>> config
    TrainingConfig(output_dir='outputs/my_model_from_script', batch_size=200, max_epochs=2, learning_rate=0.001, train_early_stopping=50, eval_early_stopping=None, steps_saving=1000, seed=8, no_cuda=False, verbose=True)

where the ``.json`` that was parsed should look like

.. code-block:: bash

    $ cat scripts/configs/training_config.json
    {"output_dir": "outputs/my_model_from_script", "batch_size": 200, "max_epochs": 2 "learning_rate": 1e-3, "train_early_stopping": 50, "eval_early_stopping": null, "steps_saving": 1000, "seed": 8, "no_cuda": false, "verbose": true}



You must ensure that the keys provided on the ``.json`` config file match the one in the required ``dataclass`` and that the value has the required tpe. 

For instance, if you want to provide your own ``training_config.json`` in the pyraug scripts, ensure that the keys in the ``.json`` file match the on in 
:class:`~pyraug.trainers.training_config.TrainingConfig` with values having the correct type. See `type checking`_. The provided scripts will indeed load ``dataclasses`` from the provided ``.json`` files.


Writing a ``.json`` from a :class:`~pyraug.config.BaseConfig` instance.
==================================================================================================


You can also write a ``.json`` directly from your ``dataclass`` using the :class:`~pyraug.config.BaseConfig.save_json` method from :class:`~pyraug.config.BaseConfig`

.. code-block:: python

    >>> from pyraug.trainers.training_config import TrainingConfig
    >>> config = TrainingConfig(max_epochs=10, learning_rate=0.1)
    >>> config.save_json(dir_path='.', filename='test')

The resulting ``.json`` should looks like this

.. code-block:: 

    $ cat test.json
    {"output_dir": null, "batch_size": 50, "max_epochs": 10, "learning_rate": 0.1, "train_early_stopping": 50, "eval_early_stopping": null, "steps_saving": 1000, "seed": 8, "no_cuda": false, "verbose": true}


.. _type checking:

Configuration type checking
=================================================


A type check is performed automatically when building the ``dataclasses`` with `pydantic <https://pydantic-docs.helpmanual.io/usage/dataclasses/>`_. Hence, if you provide the wrong type in the config it will either:

- be converted to the required type:

.. code-block:: 

    >>> from pyraug.trainers.training_config import TrainingConfig
    >>> config = TrainingConfig(max_epochs='10')
    >>> config.max_epochs, type(config.max_epochs)
    (10, <class 'int'>)

- or raise a Validation Error

.. code-block:: python


    >>> from pyraug.trainers.training_config import TrainingConfig
    >>> config = TrainingConfig(max_epochs='10_')
    Traceback (most recent call last):
        File "<stdin>", line 1, in <module>
        File "<string>", line 13, in __init__
        File "pydantic/dataclasses.py", line 99, in pydantic.dataclasses._generate_pydantic_post_init._pydantic_post_init
        # +=======+=======+=======+
    pydantic.error_wrappers.ValidationError: 1 validation error for TrainingConfig
    max_epochs
        value is not a valid integer (type=type_error.integer)


A similar check is performed on the ``.json`` when the classmethod :class:`~pyraug.config.BaseConfig.from_json_file` is called.





.. _model-setting:

************************************************
The model parameters
************************************************

Each model coded in Pyraug requires a :class:`ModelConfig` inheriting from :class:`~pyraug.models.base.base_config.BaseModelConfig` class to be built. Hence, to build a basic model you need to run the following



.. code-block:: python

    >>> from pyraug.models.my_model.my_model_config import MyModelConfig
    >>> from pyraug.models.my_model.my_model import MyModel
    >>> config = MyModelConfig(
    ...    input_dim=10 # Setting the data input dimension is needed if you do not use your own autoencoding architecture
    ...    # your parameters go here
    ... )
    >>> m = MyModel(model_config=config) # Built the model


Let now say that you want to override the model default parameters. The only thing you have to do is to pass you arguments to the ``dataclass`` :class:`ModelConfig`.


Let say, we want to change the temperature T in the metric in a :class:`~pyraug.models.RHVAE` model which defaults to 1.5 and raise it to 2. Well simply run the following.


.. code-block:: python

    >>> from pyraug.models.rhvae.rhvae_config import RHVAEConfig
    >>> from pyraug.models import RHVAE
    >>> config = RHVAEConfig(input_dim=10, temperature=2)
    >>> m = RHVAE(model_config=config)
    >>> m.temperature
    Parameter containing:
    tensor([2.])


Check out the documentation to see the whole list of parameter you can amend.

.. _trainer-setting:


************************************************
The sampler parameters
************************************************

To generate from a Pyraug's model a :class:`ModelSampler` inheriting from :class:`~pyraug.models.base.base_sampler.BaseSampler` is used. A :class:`ModelSampler` is instantiated with a Pyraug's model and a :class:`ModelSamplerConfig`. Hence, likewise the VAE models, the sampler parameters can be easily amended as follows   

.. code-block:: python

    >>> from pyraug.models.my_model.my_model_config import MyModelSamplerConfig
    >>> from pyraug.models.my_model.my_model_sampler import MyModelSampler
    >>> config = MyModelSamplerConfig(
    ...    # your parameters go here
    ... )
    >>> m = MyModelSample(model=my_model, sampler_config=config) # Built the model


Let now say that you want to override the sampler default parameters. The only thing you have to do is to pass you arguments to the ``dataclass`` :class:`ModelSamplerConfig`.

Let say, we want to change the number of leapfrog steps in the :class:`~pyraug.models.rhvae.RHVAESampler` config model which defaults to 15 and make it to 5. Well your code should look like the following.


.. code-block:: python

        >>> from pyraug.models import RHVAE
        >>> from pyraug.models.rhvae import RHVAESampler, RHVAESamplerConfig, RHVAEConfig
        >>> custom_sampler_config = RHVAESamplerConfig(
        ...   n_lf=5
        ... ) # Set up sampler config
        >>> custom_sampler = RHVAESampler(
        ...     model=model, sampler_config=custom_sampler_config
        ... ) # Build sampler
        >>> custom_sampler.n_lf
        ... tensor([5])

Check out the documentation to see the whole list of parameter you can amend.

.. _trainer-setting:

************************************************
The :class:`~pyraug.trainers.Trainer` parameters
************************************************

Likewise the VAE models, the instance :class:`~pyraug.trainers.Trainer` can be created with default parameters or you can easily amend them the same way it is done for the models. 


Example:
~~~~~~~~

Say you want to train your model for 10 epochs, with no early stopping on the train et and a learning_rate of 0.1

.. code-block:: python

    >>> from pyraug.trainers.training_config import TrainingConfig
    >>> config = TrainingConfig(
    ...    max_epochs=10, learning_rate=0.1, train_early_stopping=None)
    >>> config
    TrainingConfig(output_dir=None, batch_size=50, max_epochs=10, learning_rate=0.1, train_early_stopping=None, eval_early_stopping=None, steps_saving=1000, seed=8, no_cuda=False, verbose=True)


You can find a comprehensive description of any parameters of the :class:`~pyraug.trainers.Trainer` you can set in :class:`~pyraug.trainers.training_config.TrainingConfig`