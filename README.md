<p align="center">
	<a href="https://pypi.org/project/pythae/">
	    <img src='https://badge.fury.io/py/pythae.svg' alt='Python' />
	</a>
    <a>
	    <img src='https://img.shields.io/badge/python-3.7%7C3.8%7C3.9%2B-blueviolet' alt='Python' />
	</a>
	<a href='https://pythae.readthedocs.io/en/latest/?badge=latest'>
    	<img src='https://readthedocs.org/projects/pythae/badge/?version=latest' alt='Documentation Status' />
	</a>
	<a href='https://opensource.org/licenses/Apache-2.0'>
	    <img src='https://img.shields.io/github/license/clementchadebec/benchmark_VAE?color=blue' />
	</a><br>
    <a>
	    <img src='https://img.shields.io/badge/code%20style-black-black' />
	</a>
	<a href="https://codecov.io/gh/clementchadebec/benchmark_VAE">
  		<img src="https://codecov.io/gh/clementchadebec/benchmark_VAE/branch/main/graph/badge.svg?token=KEM7KKISXJ"/>
	</a>
	<a href="https://colab.research.google.com/github/clementchadebec/benchmark_VAE/blob/main/examples/notebooks/overview_notebook.ipynb">
  		<img src="https://colab.research.google.com/assets/colab-badge.svg"/>
	</a>
	</a>
</p>

</p>
<p align="center">
  <a href="https://pythae.readthedocs.io/en/latest/">Documentation</a>
</p>
	
    
# pythae 

This library implements some of the most common (Variational) Autoencoder models under a unified implementation. In particular, it 
provides the possibility to perform benchmark experiments and comparisons by training 
the models with the same autoencoding neural network architecture. The feature *make your own autoencoder* 
allows you to train any of these models with your own data and own Encoder and Decoder neural networks. It integrates experiment monitoring tools such [wandb](https://wandb.ai/), [mlflow](https://mlflow.org/) or [comet-ml](https://www.comet.com/signup?utm_source=pythae&utm_medium=partner&utm_campaign=AMS_US_EN_SNUP_Pythae_Comet_Integration) 🧪 and allows model sharing and loading from the [HuggingFace Hub](https://huggingface.co/models) 🤗 in a few lines of code.

**News** 📢

As of v0.1.0, `Pythae` now supports distributed training using PyTorch's [DDP](https://pytorch.org/docs/stable/notes/ddp.html). You can now train your favorite VAE faster and on larger datasets, still with a few lines of code.
See our speed-up [benchmark](#benchmark).

## Quick access:
- [Installation](#installation)
- [Implemented models](#available-models) / [Implemented samplers](#available-samplers)
- [Reproducibility statement](#reproducibility) / [Results flavor](#results)
- [Model training](#launching-a-model-training) / [Data generation](#launching-data-generation) / [Custom network architectures](#define-you-own-autoencoder-architecture) / [Distributed training](#distributed-training-with-pythae)
- [Model sharing with 🤗 Hub](#sharing-your-models-with-the-huggingface-hub-) / [Experiment tracking with `wandb`](#monitoring-your-experiments-with-wandb-) / [Experiment tracking with `mlflow`](#monitoring-your-experiments-with-mlflow-) / [Experiment tracking with `comet_ml`](#monitoring-your-experiments-with-comet_ml-)
- [Tutorials](#getting-your-hands-on-the-code) / [Documentation](https://pythae.readthedocs.io/en/latest/)
- [Contributing 🚀](#contributing-) / [Issues 🛠️](#dealing-with-issues-%EF%B8%8F)
- [Citing this repository](#citation)

# Installation

To install the latest stable release of this library run the following using ``pip``

```bash
$ pip install pythae
``` 

To install the latest github version of this library run the following using ``pip``

```bash
$ pip install git+https://github.com/clementchadebec/benchmark_VAE.git
``` 

or alternatively you can clone the github repo to access to tests, tutorials and scripts.
```bash
$ git clone https://github.com/clementchadebec/benchmark_VAE.git
```
and install the library
```bash
$ cd benchmark_VAE
$ pip install -e .
``` 

## Available Models

Below is the list of the models currently implemented in the library.


|               Models               |                                                                                    Training example                                                                                    |                     Paper                    |                           Official Implementation                          |
|:----------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------:|:--------------------------------------------------------------------------:|
| Autoencoder (AE)                   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/clementchadebec/benchmark_VAE/blob/main/examples/notebooks/models_training/ae_training.ipynb) |                                              |                                                                            |
| Variational Autoencoder (VAE)      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/clementchadebec/benchmark_VAE/blob/main/examples/notebooks/models_training/vae_training.ipynb) | [link](https://arxiv.org/abs/1312.6114)  |
| Beta Variational Autoencoder (BetaVAE) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/clementchadebec/benchmark_VAE/blob/main/examples/notebooks/models_training/beta_vae_training.ipynb) | [link](https://openreview.net/pdf?id=Sy2fzU9gl)  |   
VAE with Linear Normalizing Flows (VAE_LinNF) |  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/clementchadebec/benchmark_VAE/blob/main/examples/notebooks/models_training/vae_lin_nf_training.ipynb) | [link](https://arxiv.org/abs/1505.05770) |         
VAE with Inverse Autoregressive Flows (VAE_IAF) |  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/clementchadebec/benchmark_VAE/blob/main/examples/notebooks/models_training/vae_iaf_training.ipynb) | [link](https://arxiv.org/abs/1606.04934) |  [link](https://github.com/openai/iaf)                                  |
| Disentangled Beta Variational Autoencoder (DisentangledBetaVAE) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/clementchadebec/benchmark_VAE/blob/main/examples/notebooks/models_training/disentangled_beta_vae_training.ipynb) | [link](https://arxiv.org/abs/1804.03599)  |   
| Disentangling by Factorising (FactorVAE) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/clementchadebec/benchmark_VAE/blob/main/examples/notebooks/models_training/factor_vae_training.ipynb) | [link](https://arxiv.org/abs/1802.05983)  |                                                                            |
| Beta-TC-VAE (BetaTCVAE) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/clementchadebec/benchmark_VAE/blob/main/examples/notebooks/models_training/beta_tc_vae_training.ipynb) | [link](https://arxiv.org/abs/1802.04942)  |  [link](https://github.com/rtqichen/beta-tcvae)
| Importance Weighted Autoencoder (IWAE) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/clementchadebec/benchmark_VAE/blob/main/examples/notebooks/models_training/iwae_training.ipynb) | [link](https://arxiv.org/abs/1509.00519v4)  | [link](https://github.com/yburda/iwae)  
| Multiply Importance Weighted Autoencoder (MIWAE) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/clementchadebec/benchmark_VAE/blob/main/examples/notebooks/models_training/miwae_training.ipynb) | [link](https://arxiv.org/abs/1802.04537)  |       
| Partially Importance Weighted Autoencoder (PIWAE) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/clementchadebec/benchmark_VAE/blob/main/examples/notebooks/models_training/piwae_training.ipynb) | [link](https://arxiv.org/abs/1802.04537)  |       
| Combination Importance Weighted Autoencoder (CIWAE) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/clementchadebec/benchmark_VAE/blob/main/examples/notebooks/models_training/ciwae_training.ipynb) | [link](https://arxiv.org/abs/1802.04537)  |                                                                             |
| VAE with perceptual metric similarity (MSSSIM_VAE)      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/clementchadebec/benchmark_VAE/blob/main/examples/notebooks/models_training/ms_ssim_vae_training.ipynb) | [link](https://arxiv.org/abs/1511.06409)  |
| Wasserstein Autoencoder (WAE)      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/clementchadebec/benchmark_VAE/blob/main/examples/notebooks/models_training/wae_training.ipynb) | [link](https://arxiv.org/abs/1711.01558) | [link](https://github.com/tolstikhin/wae)                                  |
| Info Variational Autoencoder (INFOVAE_MMD)      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/clementchadebec/benchmark_VAE/blob/main/examples/notebooks/models_training/info_vae_training.ipynb) | [link](https://arxiv.org/abs/1706.02262) |                                   |
| VAMP Autoencoder (VAMP)            | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/clementchadebec/benchmark_VAE/blob/main/examples/notebooks/models_training/vamp_training.ipynb) | [link](https://arxiv.org/abs/1705.07120) | [link](https://github.com/jmtomczak/vae_vampprior)                         |
| Hyperspherical VAE (SVAE)            | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/clementchadebec/benchmark_VAE/blob/main/examples/notebooks/models_training/svae_training.ipynb) | [link](https://arxiv.org/abs/1804.00891) | [link](https://github.com/nicola-decao/s-vae-pytorch)
| Poincaré Disk VAE (PoincareVAE)            | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/clementchadebec/benchmark_VAE/blob/main/examples/notebooks/models_training/pvae_training.ipynb) | [link](https://arxiv.org/abs/1901.06033) | [link](https://github.com/emilemathieu/pvae)                         |
| Adversarial Autoencoder (Adversarial_AE)                   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/clementchadebec/benchmark_VAE/blob/main/examples/notebooks/models_training/adversarial_ae_training.ipynb) | [link](https://arxiv.org/abs/1511.05644)
| Variational Autoencoder GAN (VAEGAN) 🥗 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/clementchadebec/benchmark_VAE/blob/main/examples/notebooks/models_training/vaegan_training.ipynb) | [link](https://arxiv.org/abs/1512.09300) | [link](https://github.com/andersbll/autoencoding_beyond_pixels)| [link](https://arxiv.org/abs/1512.09300) | [link](https://github.com/andersbll/autoencoding_beyond_pixels)
| Vector Quantized VAE (VQVAE) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/clementchadebec/benchmark_VAE/blob/main/examples/notebooks/models_training/vqvae_training.ipynb) | [link](https://arxiv.org/abs/1711.00937) | [link](https://github.com/deepmind/sonnet/blob/v2/sonnet/)
| Hamiltonian VAE (HVAE)             | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/clementchadebec/benchmark_VAE/blob/main/examples/notebooks/models_training/hvae_training.ipynb) | [link](https://arxiv.org/abs/1805.11328) | [link](https://github.com/anthonycaterini/hvae-nips)                       |
| Regularized AE with L2 decoder param (RAE_L2) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/clementchadebec/benchmark_VAE/blob/main/examples/notebooks/models_training/rae_l2_training.ipynb) | [link](https://arxiv.org/abs/1903.12436) | [link](https://github.com/ParthaEth/Regularized_autoencoders-RAE-/tree/master/) |
| Regularized AE with gradient penalty (RAE_GP) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/clementchadebec/benchmark_VAE/blob/main/examples/notebooks/models_training/rae_gp_training.ipynb) | [link](https://arxiv.org/abs/1903.12436) | [link](https://github.com/ParthaEth/Regularized_autoencoders-RAE-/tree/master/) |
| Riemannian Hamiltonian VAE (RHVAE) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/clementchadebec/benchmark_VAE/blob/main/examples/notebooks/models_training/rhvae_training.ipynb) | [link](https://arxiv.org/abs/2105.00026) | [link](https://github.com/clementchadebec/pyraug)|

**See [reconstruction](#Reconstruction) and [generation](#Generation) results for all aforementionned models**

## Available Samplers

Below is the list of the models currently implemented in the library.

|                Samplers               |   Models  		  | Paper 											  | Official Implementation 				  |
|:-------------------------------------:|:-------------------:|:-------------------------------------------------:|:-----------------------------------------:|
| Normal prior (NormalSampler)                         | all models		  | [link](https://arxiv.org/abs/1312.6114)		  |
| Gaussian mixture (GaussianMixtureSampler) | all models		  | [link](https://arxiv.org/abs/1903.12436) 	  | [link](https://github.com/ParthaEth/Regularized_autoencoders-RAE-/tree/master/models/rae) |
| Two stage VAE sampler (TwoStageVAESampler)					| all VAE based models| [link](https://openreview.net/pdf?id=B1e0X3C9tQ)  | [link](https://github.com/daib13/TwoStageVAE/) |)
| Unit sphere uniform sampler (HypersphereUniformSampler)                     |    SVAE  		  | [link](https://arxiv.org/abs/1804.00891)      |		[link](https://github.com/nicola-decao/s-vae-pytorch)
| Poincaré Disk sampler (PoincareDiskSampler)                     |    PoincareVAE  		  | [link](https://arxiv.org/abs/1901.06033)      |		[link](https://github.com/emilemathieu/pvae)
| VAMP prior sampler (VAMPSampler)                   |    VAMP   		  | [link](https://arxiv.org/abs/1705.07120) 	  | [link](https://github.com/jmtomczak/vae_vampprior) |
| Manifold sampler (RHVAESampler)                     |    RHVAE  		  | [link](https://arxiv.org/abs/2105.00026)      |	[link](https://github.com/clementchadebec/pyraug)|
| Masked Autoregressive Flow Sampler (MAFSampler) | all models | [link](https://arxiv.org/abs/1705.07057v4)      |	[link](https://github.com/gpapamak/maf) |
| Inverse Autoregressive Flow Sampler (IAFSampler) | all models | [link](https://arxiv.org/abs/1606.04934) |  [link](https://github.com/openai/iaf)             |   
| PixelCNN (PixelCNNSampler) | VQVAE | [link](https://arxiv.org/abs/1606.05328) |             |                     

## Reproducibility

We validate the implementations by reproducing some results presented in the original publications when the official code has been released or when enough details about the experimental section of the papers were available. See [reproducibility](https://github.com/clementchadebec/benchmark_VAE/tree/main/examples/scripts/reproducibility) for more details.

## Launching a model training

To launch a model training, you only need to call a `TrainingPipeline` instance. 

```python
>>> from pythae.pipelines import TrainingPipeline
>>> from pythae.models import VAE, VAEConfig
>>> from pythae.trainers import BaseTrainerConfig

>>> # Set up the training configuration
>>> my_training_config = BaseTrainerConfig(
...	output_dir='my_model',
...	num_epochs=50,
...	learning_rate=1e-3,
...	per_device_train_batch_size=200,
...	per_device_eval_batch_size=200,
...	train_dataloader_num_workers=2,
...	eval_dataloader_num_workers=2,
...	steps_saving=20,
...	optimizer_cls="AdamW",
...	optimizer_params={"weight_decay": 0.05, "betas": (0.91, 0.995)},
...	scheduler_cls="ReduceLROnPlateau",
...	scheduler_params={"patience": 5, "factor": 0.5}
... )
>>> # Set up the model configuration 
>>> my_vae_config = model_config = VAEConfig(
...	input_dim=(1, 28, 28),
...	latent_dim=10
... )
>>> # Build the model
>>> my_vae_model = VAE(
...	model_config=my_vae_config
... )
>>> # Build the Pipeline
>>> pipeline = TrainingPipeline(
... 	training_config=my_training_config,
... 	model=my_vae_model
... )
>>> # Launch the Pipeline
>>> pipeline(
...	train_data=your_train_data, # must be torch.Tensor, np.array or torch datasets
...	eval_data=your_eval_data # must be torch.Tensor, np.array or torch datasets
... )
```

At the end of training, the best model weights, model configuration and training configuration are stored in a `final_model` folder available in  `my_model/MODEL_NAME_training_YYYY-MM-DD_hh-mm-ss` (with `my_model` being the `output_dir` argument of the `BaseTrainerConfig`). If you further set the `steps_saving` argument to a certain value, folders named `checkpoint_epoch_k` containing the best model weights, optimizer, scheduler, configuration and training configuration at epoch *k* will also appear in `my_model/MODEL_NAME_training_YYYY-MM-DD_hh-mm-ss`.

## Launching a training on benchmark datasets
We also provide a training script example [here](https://github.com/clementchadebec/benchmark_VAE/tree/main/examples/scripts/training.py) that can be used to train the models on benchmarks datasets (mnist, cifar10, celeba ...). The script can be launched with the following commandline

```bash
python training.py --dataset mnist --model_name ae --model_config 'configs/ae_config.json' --training_config 'configs/base_training_config.json'
```

See [README.md](https://github.com/clementchadebec/benchmark_VAE/tree/main/examples/scripts/README.md) for further details on this script

## Launching data generation

### Using the `GenerationPipeline`

The easiest way to launch a data generation from a trained model consists in using the built-in `GenerationPipeline` provided in Pythae. Say you want to generate 100 samples using a `MAFSampler` all you have to do is 1) relaod the trained model, 2) define the sampler's configuration and 3) create and launch the `GenerationPipeline` as follows

```python
>>> from pythae.models import AutoModel
>>> from pythae.samplers import MAFSamplerConfig
>>> from pythae.pipelines import GenerationPipeline
>>> # Retrieve the trained model
>>> my_trained_vae = AutoModel.load_from_folder(
...	'path/to/your/trained/model'
... )
>>> my_sampler_config = MAFSamplerConfig(
...	n_made_blocks=2,
...	n_hidden_in_made=3,
...	hidden_size=128
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
```

### Using the Samplers

Alternatively, you can launch the data generation process from a trained model directly with the sampler. For instance, to generate new data with your sampler, run the following.

```python
>>> from pythae.models import AutoModel
>>> from pythae.samplers import NormalSampler
>>> # Retrieve the trained model
>>> my_trained_vae = AutoModel.load_from_folder(
...	'path/to/your/trained/model'
... )
>>> # Define your sampler
>>> my_samper = NormalSampler(
...	model=my_trained_vae
... )
>>> # Generate samples
>>> gen_data = my_samper.sample(
...	num_samples=50,
...	batch_size=10,
...	output_dir=None,
...	return_gen=True
... )
```
If you set `output_dir` to a specific path, the generated images will be saved as `.png` files named `00000000.png`, `00000001.png` ...
The samplers can be used with any model as long as it is suited. For instance, a `GaussianMixtureSampler` instance can be used to generate from any model but a `VAMPSampler` will only be usable with a `VAMP` model. Check [here](#available-samplers) to see which ones apply to your model. Be carefull that some samplers such as the `GaussianMixtureSampler` for instance may need to be fitted by calling the `fit` method before using. Below is an example for the `GaussianMixtureSampler`. 

```python
>>> from pythae.models import AutoModel
>>> from pythae.samplers import GaussianMixtureSampler, GaussianMixtureSamplerConfig
>>> # Retrieve the trained model
>>> my_trained_vae = AutoModel.load_from_folder(
...	'path/to/your/trained/model'
... )
>>> # Define your sampler
... gmm_sampler_config = GaussianMixtureSamplerConfig(
...	n_components=10
... )
>>> my_samper = GaussianMixtureSampler(
...	sampler_config=gmm_sampler_config,
...	model=my_trained_vae
... )
>>> # fit the sampler
>>> gmm_sampler.fit(train_dataset)
>>> # Generate samples
>>> gen_data = my_samper.sample(
...	num_samples=50,
...	batch_size=10,
...	output_dir=None,
...	return_gen=True
... )
```


## Define you own Autoencoder architecture 
 
Pythae provides you the possibility to define your own neural networks within the VAE models. For instance, say you want to train a Wassertstein AE with a specific encoder and decoder, you can do the following:

```python
>>> from pythae.models.nn import BaseEncoder, BaseDecoder
>>> from pythae.models.base.base_utils import ModelOutput
>>> class My_Encoder(BaseEncoder):
...	def __init__(self, args=None): # Args is a ModelConfig instance
...		BaseEncoder.__init__(self)
...		self.layers = my_nn_layers()
...		
...	def forward(self, x:torch.Tensor) -> ModelOutput:
...		out = self.layers(x)
...		output = ModelOutput(
...			embedding=out # Set the output from the encoder in a ModelOutput instance 
...		)
...		return output
...
... class My_Decoder(BaseDecoder):
...	def __init__(self, args=None):
...		BaseDecoder.__init__(self)
...		self.layers = my_nn_layers()
...		
...	def forward(self, x:torch.Tensor) -> ModelOutput:
...		out = self.layers(x)
...		output = ModelOutput(
...			reconstruction=out # Set the output from the decoder in a ModelOutput instance
...		)
...		return output
...
>>> my_encoder = My_Encoder()
>>> my_decoder = My_Decoder()
```

And now build the model

```python
>>> from pythae.models import WAE_MMD, WAE_MMD_Config
>>> # Set up the model configuration 
>>> my_wae_config = model_config = WAE_MMD_Config(
...	input_dim=(1, 28, 28),
...	latent_dim=10
... )
...
>>> # Build the model
>>> my_wae_model = WAE_MMD(
...	model_config=my_wae_config,
...	encoder=my_encoder, # pass your encoder as argument when building the model
...	decoder=my_decoder # pass your decoder as argument when building the model
... )
```

**important note 1**: For all AE-based models (AE, WAE, RAE_L2, RAE_GP), both the encoder and decoder must return a `ModelOutput` instance. For the encoder, the `ModelOutput` instance must contain the embbeddings under the key `embedding`. For the decoder, the `ModelOutput` instance must contain the reconstructions under the key `reconstruction`.


**important note 2**: For all VAE-based models (VAE, BetaVAE, IWAE, HVAE, VAMP, RHVAE), both the encoder and decoder must return a `ModelOutput` instance. For the encoder, the `ModelOutput` instance must contain the embbeddings and **log**-covariance matrices (of shape batch_size x latent_space_dim) respectively under the key `embedding` and `log_covariance` key. For the decoder, the `ModelOutput` instance must contain the reconstructions under the key `reconstruction`.


## Using benchmark neural nets
You can also find predefined neural network architectures for the most common data sets (*i.e.* MNIST, CIFAR, CELEBA ...) that can be loaded as follows

```python
>>> from pythae.models.nn.benchmark.mnist import (
...	Encoder_Conv_AE_MNIST, # For AE based model (only return embeddings)
...	Encoder_Conv_VAE_MNIST, # For VAE based model (return embeddings and log_covariances)
...	Decoder_Conv_AE_MNIST
... )
```
Replace *mnist* by cifar or celeba to access to other neural nets.

## Distributed Training with `Pythae`
As of `v0.1.0`, Pythae now supports distributed training using PyTorch's [DDP](https://pytorch.org/docs/stable/notes/ddp.html). It allows you to train your favorite VAE faster and on larger dataset using multi-gpu and/or multi-node training.

To do so, you can build a python script that will then be launched by a launcher (such as `srun` on a cluster). The only thing that is needed in the script is to specify some elements relative to the distributed environment (such as the number of nodes/gpus) directly in the training configuration as follows

```python
>>> training_config = BaseTrainerConfig(
...     num_epochs=10,
...     learning_rate=1e-3,
...     per_device_train_batch_size=64,
...     per_device_eval_batch_size=64,
...     train_dataloader_num_workers=8,
...     eval_dataloader_num_workers=8,
...     dist_backend="nccl", # distributed backend
...     world_size=8 # number of gpus to use (n_nodes x n_gpus_per_node),
...     rank=5 # process/gpu id,
...     local_rank=1 # node id,
...     master_addr="localhost" # master address,
...     master_port="12345" # master port,
... )
```

See this [example script](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/scripts/distributed_training_imagenet.py) that defines a multi-gpu VQVAE training on ImageNet dataset. Please note that the way the distributed environnement variables (`world_size`, `rank` ...) are recovered may be specific to the cluster and launcher you use. 

### Benchmark

Below are indicated the training times for a Vector Quantized VAE (VQ-VAE) with `Pythae` for 100 epochs on MNIST on V100 16GB GPU(s), for 50 epochs on [FFHQ](https://github.com/NVlabs/ffhq-dataset) (1024x1024 images) and for 20 epochs on [ImageNet-1k](https://huggingface.co/datasets/imagenet-1k) on V100 32GB GPU(s).

|  | Train Data | 1 GPU | 4 GPUs | 2x4 GPUs |
|:---:|:---:|:---:|:---:|---|
| MNIST (VQ-VAE) | 28x28 images (50k) | 235.18 s | 62.00 s | 35.86 s |
| FFHQ 1024x1024 (VQVAE) | 1024x1024 RGB images (60k) | 19h 1min | 5h 6min | 2h 37min |
| ImageNet-1k 128x128 (VQVAE) | 128x128 RGB images (~ 1.2M) | 6h 25min | 1h 41min | 51min 26s |


For each dataset, we provide the benchmarking scripts [here](https://github.com/clementchadebec/benchmark_VAE/tree/main/examples/scripts)


## Sharing your models with the HuggingFace Hub 🤗
Pythae also allows you to share your models on the [HuggingFace Hub](https://huggingface.co/models). To do so you need:
- a valid HuggingFace account
- the package `huggingface_hub` installed in your virtual env. If not you can install it with 
```
$ python -m pip install huggingface_hub
```
- to be logged in to your HuggingFace account using
```
$ huggingface-cli login
```

### Uploading a model to the Hub
Any pythae model can be easily uploaded using the method `push_to_hf_hub`
```python
>>> my_vae_model.push_to_hf_hub(hf_hub_path="your_hf_username/your_hf_hub_repo")
```
**Note:** If `your_hf_hub_repo` already exists and is not empty, files will be overridden. In case, 
the repo `your_hf_hub_repo` does not exist, a folder having the same name will be created.

### Downloading models from the Hub
Equivalently, you can download or reload any Pythae's model directly from the Hub using the method `load_from_hf_hub`
```python
>>> from pythae.models import AutoModel
>>> my_downloaded_vae = AutoModel.load_from_hf_hub(hf_hub_path="path_to_hf_repo")
```

## Monitoring your experiments with `wandb` 🧪
Pythae also integrates the experiment tracking tool [wandb](https://wandb.ai/) allowing users to store their configs, monitor their trainings and compare runs through a graphic interface. To be able use this feature you will need:
- a valid wandb account
- the package `wandb` installed in your virtual env. If not you can install it with 
```
$ pip install wandb
```
- to be logged in to your wandb account using
```
$ wandb login
```

### Creating a `WandbCallback`
Launching an experiment monitoring with `wandb` in pythae is pretty simple. The only thing a user needs to do is create a `WandbCallback` instance...

```python
>>> # Create you callback
>>> from pythae.trainers.training_callbacks import WandbCallback
>>> callbacks = [] # the TrainingPipeline expects a list of callbacks
>>> wandb_cb = WandbCallback() # Build the callback 
>>> # SetUp the callback 
>>> wandb_cb.setup(
...	training_config=your_training_config, # training config
...	model_config=your_model_config, # model config
...	project_name="your_wandb_project", # specify your wandb project
...	entity_name="your_wandb_entity", # specify your wandb entity
... )
>>> callbacks.append(wandb_cb) # Add it to the callbacks list
```
...and then pass it to the `TrainingPipeline`.
```python
>>> pipeline = TrainingPipeline(
...	training_config=config,
...	model=model
... )
>>> pipeline(
...	train_data=train_dataset,
...	eval_data=eval_dataset,
...	callbacks=callbacks # pass the callbacks to the TrainingPipeline and you are done!
... )
>>> # You can log to https://wandb.ai/your_wandb_entity/your_wandb_project to monitor your training
```
See the detailed tutorial 

## Monitoring your experiments with `mlflow` 🧪
Pythae also integrates the experiment tracking tool [mlflow](https://mlflow.org/) allowing users to store their configs, monitor their trainings and compare runs through a graphic interface. To be able use this feature you will need:
- the package `mlfow` installed in your virtual env. If not you can install it with 
```
$ pip install mlflow
```

### Creating a `MLFlowCallback`
Launching an experiment monitoring with `mlfow` in pythae is pretty simple. The only thing a user needs to do is create a `MLFlowCallback` instance...

```python
>>> # Create you callback
>>> from pythae.trainers.training_callbacks import MLFlowCallback
>>> callbacks = [] # the TrainingPipeline expects a list of callbacks
>>> mlflow_cb = MLFlowCallback() # Build the callback 
>>> # SetUp the callback 
>>> mlflow_cb.setup(
...	training_config=your_training_config, # training config
...	model_config=your_model_config, # model config
...	run_name="mlflow_cb_example", # specify your mlflow run
... )
>>> callbacks.append(mlflow_cb) # Add it to the callbacks list
```
...and then pass it to the `TrainingPipeline`.
```python
>>> pipeline = TrainingPipeline(
...	training_config=config,
...	model=model
... )
>>> pipeline(
...	train_data=train_dataset,
...	eval_data=eval_dataset,
...	callbacks=callbacks # pass the callbacks to the TrainingPipeline and you are done!
... )
```
you can visualize your metric by running the following in the directory where the `./mlruns`
```bash
$ mlflow ui 
```
See the detailed tutorial 

## Monitoring your experiments with `comet_ml` 🧪
Pythae also integrates the experiment tracking tool [comet_ml](https://www.comet.com/signup?utm_source=pythae&utm_medium=partner&utm_campaign=AMS_US_EN_SNUP_Pythae_Comet_Integration) allowing users to store their configs, monitor their trainings and compare runs through a graphic interface. To be able use this feature you will need:
- the package `comet_ml` installed in your virtual env. If not you can install it with 
```
$ pip install comet_ml
```

### Creating a `CometCallback`
Launching an experiment monitoring with `comet_ml` in pythae is pretty simple. The only thing a user needs to do is create a `CometCallback` instance...

```python
>>> # Create you callback
>>> from pythae.trainers.training_callbacks import CometCallback
>>> callbacks = [] # the TrainingPipeline expects a list of callbacks
>>> comet_cb = CometCallback() # Build the callback 
>>> # SetUp the callback 
>>> comet_cb.setup(
...	training_config=training_config, # training config
...	model_config=model_config, # model config
...	api_key="your_comet_api_key", # specify your comet api-key
...	project_name="your_comet_project", # specify your wandb project
...	#offline_run=True, # run in offline mode
...	#offline_directory='my_offline_runs' # set the directory to store the offline runs
... )
>>> callbacks.append(comet_cb) # Add it to the callbacks list
```
...and then pass it to the `TrainingPipeline`.
```python
>>> pipeline = TrainingPipeline(
...	training_config=config,
...	model=model
... )
>>> pipeline(
...	train_data=train_dataset,
...	eval_data=eval_dataset,
...	callbacks=callbacks # pass the callbacks to the TrainingPipeline and you are done!
... )
>>> # You can log to https://comet.com/your_comet_username/your_comet_project to monitor your training
```
See the detailed tutorial 


## Getting your hands on the code 

To help you to understand the way pythae works and how you can train your models with this library we also
provide tutorials:

- [making_your_own_autoencoder.ipynb](https://github.com/clementchadebec/benchmark_VAE/tree/main/examples/notebooks) shows you how to pass your own networks to the models implemented in pythae [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/clementchadebec/benchmark_VAE/blob/main/examples/notebooks/making_your_own_autoencoder.ipynb)

- [custom_dataset.ipynb](https://github.com/clementchadebec/benchmark_VAE/tree/main/examples/notebooks) shows you how to  use custom datasets with any of the models implemented in pythae [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/clementchadebec/benchmark_VAE/blob/main/examples/notebooks/custom_dataset.ipynb)

- [hf_hub_models_sharing.ipynb](https://github.com/clementchadebec/benchmark_VAE/tree/main/examples/notebooks) shows you how to upload and download models for the HuggingFace Hub [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/clementchadebec/benchmark_VAE/blob/main/examples/notebooks/hf_hub_models_sharing.ipynb)

- [wandb_experiment_monitoring.ipynb](https://github.com/clementchadebec/benchmark_VAE/tree/main/examples/notebooks) shows you how to monitor you experiments using `wandb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/clementchadebec/benchmark_VAE/blob/main/examples/notebooks/wandb_experiment_monitoring.ipynb)

- [mlflow_experiment_monitoring.ipynb](https://github.com/clementchadebec/benchmark_VAE/tree/main/examples/notebooks) shows you how to monitor you experiments using `mlflow` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/clementchadebec/benchmark_VAE/blob/main/examples/notebooks/mlflow_experiment_monitoring.ipynb)

- [comet_experiment_monitoring.ipynb](https://github.com/clementchadebec/benchmark_VAE/tree/main/examples/notebooks) shows you how to monitor you experiments using `comet_ml` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/clementchadebec/benchmark_VAE/blob/main/examples/notebooks/comet_experiment_monitoring.ipynb)

- [models_training](https://github.com/clementchadebec/benchmark_VAE/tree/main/examples/notebooks/models_training) folder provides notebooks showing how to train each implemented model and how to sample from it using `pythae.samplers`.

- [scripts](https://github.com/clementchadebec/benchmark_VAE/tree/main/examples/scripts) folder provides in particular an example of a training script to train the models on benchmark data sets (mnist, cifar10, celeba ...)

## Dealing with issues 🛠️

If you are experiencing any issues while running the code or request new features/models to be implemented please [open an issue on github](https://github.com/clementchadebec/benchmark_VAE/issues).

## Contributing 🚀

You want to contribute to this library by adding a model, a sampler or simply fix a bug ? That's awesome! Thank you! Please see [CONTRIBUTING.md](https://github.com/clementchadebec/benchmark_VAE/tree/main/CONTRIBUTING.md) to follow the main contributing guidelines.

## Results

### Reconstruction
First let's have a look at the reconstructed samples taken from the evaluation set. 


|               Models               |                                                                                    MNIST                                                                     |                     CELEBA             
|:----------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------:|
| Eval data                  | ![Eval](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/eval_reconstruction_mnist.png) | ![AE](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/eval_reconstruction_celeba.png)  
| AE                  | ![AE](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/ae_reconstruction_mnist.png) | ![AE](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/ae_reconstruction_celeba.png)                                                                            |
| VAE | ![VAE](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/vae_reconstruction_mnist.png) |  ![VAE](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/vae_reconstruction_celeba.png)
| Beta-VAE| ![Beta](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/beta_vae_reconstruction_mnist.png) | ![Beta Normal](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/beta_vae_reconstruction_celeba.png)
| VAE Lin NF| ![VAE_LinNF](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/vae_lin_nf_reconstruction_mnist.png) | ![VAE_IAF Normal](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/vae_lin_nf_reconstruction_celeba.png)
| VAE IAF| ![VAE_IAF](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/vae_iaf_reconstruction_mnist.png) | ![VAE_IAF Normal](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/vae_iaf_reconstruction_celeba.png)
| Disentangled  Beta-VAE| ![Disentangled Beta](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/disentangled_beta_vae_reconstruction_mnist.png) | ![Disentangled Beta](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/disentangled_beta_vae_reconstruction_celeba.png)
| FactorVAE| ![FactorVAE](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/factor_vae_reconstruction_mnist.png) | ![FactorVAE](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/factor_vae_reconstruction_celeba.png)
| BetaTCVAE| ![BetaTCVAE](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/beta_tc_vae_reconstruction_mnist.png) | ![BetaTCVAE](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/beta_tc_vae_reconstruction_celeba.png)
| IWAE | ![IWAE](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/iwae_reconstruction_mnist.png) | ![IWAE](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/iwae_reconstruction_celeba.png)
| MSSSIM_VAE | ![MSSSIM VAE](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/msssim_vae_reconstruction_mnist.png) |  ![MSSSIM VAE](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/msssim_vae_reconstruction_celeba.png)
| WAE| ![WAE](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/wae_reconstruction_mnist.png) | ![WAE](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/wae_reconstruction_celeba.png)
| INFO VAE| ![INFO](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/infovae_reconstruction_mnist.png) | ![INFO](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/infovae_reconstruction_celeba.png)
| VAMP | ![VAMP](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/vamp_reconstruction_mnist.png) | ![VAMP](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/vamp_reconstruction_celeba.png) |
| SVAE | ![SVAE](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/svae_reconstruction_mnist.png) | ![SVAE](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/svae_reconstruction_celeba.png) |
| Adversarial_AE          | ![AAE](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/aae_reconstruction_mnist.png) | ![AAE](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/aae_reconstruction_celeba.png) |
| VAE_GAN          | ![VAEGAN](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/vaegan_reconstruction_mnist.png) | ![VAEGAN](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/vaegan_reconstruction_celeba.png) |
| VQVAE          | ![VQVAE](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/vqvae_reconstruction_mnist.png) | ![VQVAE](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/vqvae_reconstruction_celeba.png) |
| HVAE             | ![HVAE](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/hvae_reconstruction_mnist.png) | ![HVAE](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/hvae_reconstruction_celeba.png)
| RAE_L2 | ![RAE L2](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/rae_l2_reconstruction_mnist.png)  |  ![RAE L2](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/rae_l2_reconstruction_celeba.png)
| RAE_GP | ![RAE GMM](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/rae_gp_reconstruction_mnist.png)  |  ![RAE GMM](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/rae_gp_reconstruction_celeba.png)
| Riemannian Hamiltonian VAE (RHVAE)| ![RHVAE](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/rhvae_reconstruction_mnist.png) | ![RHVAE RHVAE](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/rhvae_reconstruction_celeba.png)

----------------------------
### Generation

Here, we show the generated samples using each model implemented in the library and different samplers.

|               Models               |                                                                                    MNIST                                                                     |                     CELEBA             
|:----------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------:|
| AE  + GaussianMixtureSampler                  | ![AE GMM](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/ae_gmm_sampling_mnist.png) | ![AE GMM](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/ae_gmm_sampling_celeba.png)                                                                            |
| VAE  + NormalSampler    | ![VAE Normal](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/vae_normal_sampling_mnist.png) |  ![VAE Normal](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/vae_normal_sampling_celeba.png)
| VAE  + GaussianMixtureSampler    | ![VAE GMM](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/vae_gmm_sampling_mnist.png) |  ![VAE GMM](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/vae_gmm_sampling_celeba.png)
| VAE  + TwoStageVAESampler    | ![VAE 2 stage](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/vae_second_stage_sampling_mnist.png) |  ![VAE 2 stage](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/vae_second_stage_sampling_celeba.png)
| VAE  + MAFSampler    | ![VAE MAF](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/vae_maf_sampling_mnist.png) |  ![VAE MAF](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/vae_maf_sampling_celeba.png)
| Beta-VAE + NormalSampler | ![Beta Normal](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/beta_vae_normal_sampling_mnist.png) | ![Beta Normal](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/beta_vae_normal_sampling_celeba.png)
| VAE Lin NF + NormalSampler | ![VAE_LinNF Normal](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/vae_lin_nf_normal_sampling_mnist.png) | ![VAE_LinNF Normal](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/vae_lin_nf_normal_sampling_celeba.png)
| VAE IAF + NormalSampler | ![VAE_IAF Normal](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/vae_iaf_normal_sampling_mnist.png) | ![VAE IAF Normal](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/vae_iaf_normal_sampling_celeba.png)
| Disentangled Beta-VAE + NormalSampler | ![Disentangled Beta Normal](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/disentangled_beta_vae_normal_sampling_mnist.png) | ![Disentangled Beta Normal](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/disentangled_beta_vae_normal_sampling_celeba.png)
| FactorVAE + NormalSampler | ![FactorVAE Normal](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/factor_vae_normal_sampling_mnist.png) | ![FactorVAE Normal](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/factor_vae_normal_sampling_celeba.png)
| BetaTCVAE + NormalSampler | ![BetaTCVAE Normal](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/beta_tc_vae_normal_sampling_mnist.png) | ![BetaTCVAE Normal](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/beta_tc_vae_normal_sampling_celeba.png)
| IWAE +  Normal sampler | ![IWAE Normal](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/iwae_normal_sampling_mnist.png) | ![IWAE Normal](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/iwae_normal_sampling_celeba.png)
| MSSSIM_VAE  + NormalSampler    | ![MSSSIM_VAE Normal](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/msssim_vae_normal_sampling_mnist.png) |  ![MSSSIM_VAE Normal](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/msssim_vae_normal_sampling_celeba.png)
| WAE + NormalSampler| ![WAE Normal](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/wae_normal_sampling_mnist.png) | ![WAE Normal](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/wae_normal_sampling_celeba.png)
| INFO VAE + NormalSampler| ![INFO Normal](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/infovae_normal_sampling_mnist.png) | ![INFO Normal](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/infovae_normal_sampling_celeba.png)
| SVAE + HypershereUniformSampler          | ![SVAE Sphere](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/svae_hypersphere_uniform_sampling_mnist.png) | ![SVAE Sphere](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/svae_hypersphere_uniform_sampling_celeba.png) |
| VAMP + VAMPSampler          | ![VAMP Vamp](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/vamp_vamp_sampling_mnist.png) | ![VAMP Vamp](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/vamp_vamp_sampling_celeba.png) |
| Adversarial_AE + NormalSampler          | ![AAE_Normal](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/aae_normal_sampling_mnist.png) | ![AAE_Normal](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/aae_normal_sampling_celeba.png) |
| VAEGAN + NormalSampler          | ![VAEGAN_Normal](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/vaegan_normal_sampling_mnist.png) | ![VAEGAN_Normal](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/vaegan_normal_sampling_celeba.png) |
| VQVAE + MAFSampler          | ![VQVAE_MAF](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/vqvae_maf_sampling_mnist.png) | ![VQVAE_MAF](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/vqvae_maf_sampling_celeba.png) |
| HVAE + NormalSampler             | ![HVAE Normal](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/hvae_normal_sampling_mnist.png) | ![HVAE GMM](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/hvae_normal_sampling_celeba.png)
| RAE_L2 + GaussianMixtureSampler | ![RAE L2 GMM](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/rae_l2_gmm_sampling_mnist.png)  |  ![RAE L2 GMM](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/rae_l2_gmm_sampling_celeba.png)
| RAE_GP + GaussianMixtureSampler| ![RAE GMM](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/rae_gp_gmm_sampling_mnist.png)  |  ![RAE GMM](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/rae_gp_gmm_sampling_celeba.png)
| Riemannian Hamiltonian VAE (RHVAE) + RHVAE Sampler| ![RHVAE RHVAE](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/rhvae_rhvae_sampling_mnist.png) | ![RHVAE RHVAE](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/rhvae_rhvae_sampling_celeba.png)


# Citation

If you find this work useful or use it in your research, please consider citing us

```bibtex
@inproceedings{chadebec2022pythae,
 author = {Chadebec, Cl\'{e}ment and Vincent, Louis and Allassonniere, Stephanie},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {S. Koyejo and S. Mohamed and A. Agarwal and D. Belgrave and K. Cho and A. Oh},
 pages = {21575--21589},
 publisher = {Curran Associates, Inc.},
 title = {Pythae: Unifying Generative Autoencoders in Python - A Benchmarking Use Case},
 volume = {35},
 year = {2022}
}
```
