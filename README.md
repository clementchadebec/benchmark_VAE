<p align="center">
<a href='https://pypi.org/project/pythae/'>
	    <img src='https://badge.fury.io/py/pythae.svg' />
	</a>
	<a href='https://opensource.org/licenses/Apache-2.0'>
	    <img src='https://img.shields.io/github/license/clementchadebec/pythae?color=blue' />
	</a>
	<a href='https://pythae.readthedocs.io/en/latest/?badge=latest'>
	    <img src='https://readthedocs.org/projects/pythae/badge/?version=latest' alt='Documentation 	Status' />
	</a>
	<a href='https://pepy.tech/project/pythae'>
	    <img src='https://static.pepy.tech/personalized-badge/pythae?period=month&units=international_system&left_color=grey&right_color=orange&left_text=downloads' alt='Downloads 	Status' />
	</a>
</p>
<p align="center">
  <a href="https://pythae.readthedocs.io/en/latest/">Documentation</a>
</p>
	


# pythae 

This library implements some of the most common (Variational) Autoencoder models. In particular it 
provides the possibility to perform benchmark experiments by training the models with the same autoencoding 
neural architecture. The feature *make your own autoencoder* allows you to train any of these models 
with your own data and Encoder and Decoder neural networks.  


# Installation

To install the library from [pypi.org](https://pypi.org/) run the following using ``pip``

```bash
$ pip install pythae
``` 


or alternatively you can clone the github repo to access to tests, tutorials and scripts.
```bash
$ git clone https://github.com/clementchadebec/pythae.git
```
and install the library
```bash
$ cd pythae
$ pip install .
``` 

## Available Models

Below is the list of the models currently implemented in the library.


|               Models               |                                                                                    Training example                                                                                    |                     Paper                    |                           Official Implementation                          |
|:----------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------:|:--------------------------------------------------------------------------:|
| Autoencoder (AE)                   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/clementchadebec/pythae/blob/main/examples/models_training/ae_training.ipynb) |                                              |                                                                            |
| Variational Autoencoder (VAE)      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/clementchadebec/pythae/blob/main/examples/models_training/vae_training.ipynb) | [link](https://arxiv.org/pdf/1312.6114.pdf)  |
| Beta Variational Autoencoder (VAE) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/clementchadebec/pythae/blob/main/examples/models_training/beta_vae_training.ipynb) | [link](https://openreview.net/pdf?id=Sy2fzU9gl)  |                                                                            |
| Wasserstein Autoencoder (WAE)      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/clementchadebec/pythae/blob/main/examples/models_training/wae_training.ipynb) | [link](https://arxiv.org/pdf/1711.01558.pdf) | [link](https://github.com/tolstikhin/wae)                                  |
| VAMP Autoencoder (VAMP)            | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/clementchadebec/pythae/blob/main/examples/models_training/vamp_training.ipynb) | [link](https://arxiv.org/pdf/1705.07120.pdf) | [link](https://github.com/jmtomczak/vae_vampprior)                         |
| Hamiltonian VAE (HVAE)             | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/clementchadebec/pythae/blob/main/examples/models_training/hvae_training.ipynb) | [link](https://arxiv.org/pdf/1805.11328.pdf) | [link](https://github.com/anthonycaterini/hvae-nips)                       |
| Riemannian Hamiltonian VAE (RHVAE) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/clementchadebec/pythae/blob/main/examples/models_training/rhvae_training.ipynb) | [link](https://arxiv.org/pdf/2010.11518.pdf) | [link](https://github.com/clementchadebec/Data_Augmentation_with_VAE-DALI) |


## Launching a model training

To launch a model training, you only need to call a `TrainingPipeline` instance. 

```python
>>> from pythae.pipelines import TrainingPipeline
>>> from pythae.models import VAE, VAEConfig

>>> # Set up the training configuration
>>> my_training_config = TrainingConfig(
...		output_dir='my_model',
...		num_epochs=50,
...		learning_rate=1e-3,
...		batch_size=200,
...		steps_saving=None
... )
>>> # Set up the model configuration 
>>> my_vae_config = model_config = VAEConfig(
...		input_dim=(1, 28, 28),
...		latent_dim=10
... )
>>> # Build the model
>>> my_vae_model = VAE(
...		model_config=my_vae_config
... )
>>> # Build the Pipeline
>>> pipeline = TrainingPipeline(
... 	training_config=my_training_config,
... 	model=my_vae_model
...	)
>>> # Launch the Pipeline
>>> pipeline(
...		train_data=your_train_data, # must be torch.Tensor or np.array 
...		eval_data=your_eval_data # must be torch.Tensor or np.array
...	)
```

At the end of training, the best model weights, model configuration and training configuration are stored in a `final_model` folder available in  `my_model/MODEL_NAME_training_YYYY-MM-DD_hh-mm-ss` (with `my_model` being the `output_dir` argument of the `TrainingConfig`). If you further set the `steps_saving` argument to a a certain value, folders named `checkpoint_epoch_k` containing the best model weights, configuration and training configuration at epoch *k* will also appear in `my_model/MODEL_NAME_training_YYYY-MM-DD_hh-mm-ss`.

## Launching data generation

To launch the data generation process from a trained model, you only need to build you sampler and retrieve your 
Several samplers are available for each models please check here to see which ones apply to your vae.
, run the following.

```python
>>> from pythae.models import VAE
>>> from pythae.samplers import NormalSampler
>>> # Retrieve the trained model
>>> my_trained_vae = VAE.load_from_folder(
...		'path/to/your/trained/model'
...	)
>>> # Define your sampler
>>> my_samper = NormalSampler(
...		model=my_trained_vae
...	)
>>> # Generate samples
>>> gen_data = normal_samper.sample(
...		num_samples=50,
...		batch_size=10,
...		output_dir=None,
...		return_gen=True
...	)
```
If you set `output_dir` to a specific path the generated images will be saved as `.png` files named `00000000.png`, `00000001.png` ...


## Define you own Autoencoder architecture
 
Say you want to train a Wassertstein AE with a specific encoder and decoder. Pythae provides you the possibility to define your own neural networks as follows

```python
>>>	from pythae.models.nn import BaseEncoder, BaseDecoder
>>> from pythae.models.base.base_utils import ModelOuput
>>>	class My_Encoder(BaseEncoder):
...		def __init__(self, args=None): # Args is a ModelConfig instance
...			BaseEncoder.__init__(self)
...			self.layers = my_nn_layers()
...		
...		def forward(self, x:torch.Tensor) -> ModelOuput:
...			out = self.layers(x)
...			output = ModelOuput(
...				embedding=out # Set the output from the encoder in a ModelOuput instance 
...			)
...			return output
...
... class My_Decoder(BaseEncoder):
...		def __init__(self, args=None):
...			BaseEncoder.__init__(self)
...			self.layers = my_nn_layers()
...		
...		def forward(self, x:torch.Tensor) -> ModelOuput:
...			out = self.layers(x)
...			output = ModelOuput(
...				reconstruction=out # Set the output from the decoder in a ModelOuput instance
...			)
...			return output
...
>>> my_encoder = My_Encoder()
>>> my_decoder = My_Decoder()
```

And now build the model

```python
>>> from pythae.models import WAE_MMD, WAE_MMD_Config
>>> # Set up the model configuration 
>>> my_wae_config = model_config = WAE_MMD_Config(
...		input_dim=(1, 28, 28),
...		latent_dim=10
... )
...
>>> # Build the model
>>> my_wae_model = WAE_MMD(
...		model_config=my_wae_config,
...		encoder=my_encoder, # pass your encoder as argument when building the model
...		decoder=my_decoder # pass your decoder as argument when building the model
... )
```

## Using benchmark neural nets
You may also find predefined neural network architecture for the most common data sets (*i.e.* MNIST, CIFAR, CELEBA ...) that can be loaded as follows

```python
>>> for pythae.models.nn.benchmark.mnist import (
...		Encoder_AE_MNIST, # For AE based model (only return embeddings)
... 	Encoder_VAE_MNIST, # For VAE based model (return embeddings and log_covariances)
... 	Decoder_AE_MNIST
)
```
Replace *mnist* by cifar or celeba to access to other neural nets.

## Getting your hands on the code

To help you to understand the way pythae works and how you can augment your data with this library we also
provide tutorials that can be found in [examples folder](https://github.com/clementchadebec/pythae/tree/main/examples):

- [getting_started.ipynb](https://github.com/clementchadebec/pythae/tree/main/examples) explains you how to train a model and generate new data using pythae's Pipelines [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/clementchadebec/pythae/blob/main/examples/getting_started.ipynb)
- [playing_with_configs.ipynb](https://github.com/clementchadebec/pythae/tree/main/examples) shows you how to amend the predefined configuration to adapt them to you data [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/clementchadebec/pythae/blob/main/examples/playing_with_configs.ipynb)
- [making_your_own_autoencoder.ipynb](https://github.com/clementchadebec/pythae/tree/main/examples) shows you how to pass your own networks to the models implemented in pythae [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/clementchadebec/pythae/blob/main/examples/making_your_own_autoencoder.ipynb)

## Dealing with issues

If you are experiencing any issues while running the code or request new features/models to be implemented please [open an issue on github](https://github.com/clementchadebec/pythae/issues).

