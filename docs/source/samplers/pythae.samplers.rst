**********************************
Samplers
**********************************

.. toctree::
    :hidden:
    :maxdepth: 1

    basesampler
    normal_sampler
    gmm_sampler
    twostage_sampler
    unit_sphere_unif_sampler
    poincare_disk_sampler
    vamp_sampler
    rhvae_sampler
    maf_sampler
    iaf_sampler
    pixelcnn_sampler
    

.. automodule::
    pythae.samplers
    

.. autosummary::
    ~pythae.samplers.BaseSampler
    ~pythae.samplers.NormalSampler
    ~pythae.samplers.GaussianMixtureSampler
    ~pythae.samplers.TwoStageVAESampler
    ~pythae.samplers.HypersphereUniformSampler
    ~pythae.samplers.PoincareDiskSampler
    ~pythae.samplers.VAMPSampler
    ~pythae.samplers.RHVAESampler
    ~pythae.samplers.MAFSampler
    ~pythae.samplers.IAFSampler
    ~pythae.samplers.PixelCNNSampler
    :nosignatures:


Basic Examples
----------------

To launch the data generation process from a trained model, you only need to build your sampler. 
For instance, to generate new data with your sampler, run the following.

Normal sampling
~~~~~~~~~~~~~~~~~

.. code-block::

    >>> from pythae.models import VAE
    >>> from pythae.samplers import NormalSampler
    >>> # Retrieve the trained model
    >>> my_trained_vae = VAE.load_from_folder(
    ...	    'path/to/your/trained/model'
    ... )
    >>> # Define your sampler
    >>> my_samper = NormalSampler(
    ...	    model=my_trained_vae
    ... )
    >>> # Generate samples
    >>> gen_data = my_samper.sample(
    ...	    num_samples=50,
    ...	    batch_size=10,
    ...	    output_dir=None,
    ...	    return_gen=True
    ... )

Gaussian mixture sampling
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block::

    >>> from pythae.models import VAE
    >>> from pythae.samplers import GaussianMixtureSampler, GaussianMixtureSamplerConfig
    >>> # Retrieve the trained model
    >>> my_trained_vae = VAE.load_from_folder(
    ...	    'path/to/your/trained/model'
    ... )
    >>> # Define your sampler
    ...     gmm_sampler_config = GaussianMixtureSamplerConfig(
    ...	    n_components=10
    ... )
    >>> my_samper = GaussianMixtureSampler(
    ...	    sampler_config=gmm_sampler_config,
    ...	    model=my_trained_vae
    ... )
    >>> # fit the sampler
    >>> gmm_sampler.fit(train_dataset)
    >>> # Generate samples
    >>> gen_data = my_samper.sample(
    ...	    num_samples=50,
    ...	    batch_size=10,
    ...	    output_dir=None,
    ...	    return_gen=True
    ... )

See also `tutorials <https://github.com/clementchadebec/benchmark_VAE/tree/main/examples/notebooks>`_.