**********************************
Pipelines
**********************************

.. automodule::
   pythae.pipelines


.. toctree::
   :maxdepth: 1

   training
   generation


.. autosummary::
    ~pythae.pipelines.TrainingPipeline
    ~pythae.pipelines.GenerationPipeline
    :nosignatures:

Basic Examples
~~~~~~~~~~~~~~~

To launch a model training with the :class:`~pythae.pipelines.TrainingPipeline`, you only need to set up your :class:`~pythae.trainers.BaseTrainerConfig`, 
:class:`~pythae.models.BaseAEConfig` build the model and trainer accordingly and then call a :class:`~pythae.pipelines.TrainingPipeline` 
instance. 

.. code-block::

    >>> from pythae.pipelines import TrainingPipeline
    >>> from pythae.models import VAE, VAEConfig
    >>> from pythae.trainers import BaseTrainerConfig

    >>> # Set up the training configuration
    >>> my_training_config = BaseTrainerConfig(
    ...	    output_dir='my_model',
    ...	    num_epochs=50,
    ...	    learning_rate=1e-3,
    ...	    per_device_train_batch_size=64,
    ...      per_device_eval_batch_size=64,
    ...	    steps_saving=None
    ... )
    >>> # Set up the model configuration 
    >>> my_vae_config = model_config = VAEConfig(
    ...	    input_dim=(1, 28, 28),
    ...	    latent_dim=10
    ... )
    >>> # Build the model
    >>> my_vae_model = VAE(
    ...	    model_config=my_vae_config
    ... )
    >>> # Build the Pipeline
    >>> pipeline = TrainingPipeline(
    ...     training_config=my_training_config,
    ...     model=my_vae_model
    ... )
    >>> # Launch the Pipeline
    >>> pipeline(
    ...	    train_data=your_train_data, # must be torch.Tensor or np.array 
    ...	    eval_data=your_eval_data # must be torch.Tensor or np.array
    ... )




To launch a data generation from a trained model using the :class:`~pythae.pipelines.GenerationPipeline`
provided in Pythae you only need 1) a trained model, 2) the sampler's configuration and 3) to create and launch the pipeline as follows

.. code-block::

   >>> from pythae.models import AutoModel
   >>> from pythae.samplers import MAFSamplerConfig
   >>> from pythae.pipelines import GenerationPipeline
   >>> # Retrieve the trained model
   >>> my_trained_vae = AutoModel.load_from_folder(
   ...	'path/to/your/trained/model'
   ... )
   >>> my_sampler_config = MAFSamplerConfig(
   ...	n_made_blocks: int = 2
   ...	n_hidden_in_made: int = 3
   ...	hidden_size: int = 128
   ... )
   >>> # Build the pipeline
   >>> pipe = GenerationPipeline(
   ...	model=my_trained_vae,
   ...	sampler_config=my_sampler_config
   ... )
   >>> # Launch data generation
   >>> generated_samples = pipe(
   ...	num_samples=args.num_samples,
   ...	return_gen=True, # If false returns nothing
   ...	train_data=train_data, # Needed to fit the sampler
   ...	eval_data=eval_data, # Needed to fit the sampler
   ...	training_config=BaseTrainerConfig(num_epochs=200) # TrainingConfig to use to fit the sampler
   ... )