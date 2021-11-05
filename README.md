<p align="center">
    <a>
	    <img src='https://img.shields.io/badge/python-3.6%7C3.7%7C3.8-blueviolet' alt='Python' />
	</a>
	<a href='https://opensource.org/licenses/Apache-2.0'>
	    <img src='https://img.shields.io/github/license/clementchadebec/benchmark_VAE?color=blue' />
	</a>
    <a>
	    <img src='https://img.shields.io/badge/code%20style-black-black' />
	</a>
	<a href="https://codecov.io/gh/clementchadebec/benchmark_VAE">
  <img src="https://codecov.io/gh/clementchadebec/benchmark_VAE/branch/main/graph/badge.svg?token=KEM7KKISXJ"/>
</a>
	</a>
</p>

      
    
# pythae 

This library implements some of the most common (Variational) Autoencoder models. In particular it 
provides the possibility to perform benchmark experiments and comparison by training 
the models with the same autoencoding neural architecture. The feature *make your own autoencoder* 
allows you to train any of these models with your own data and own Encoder and Decoder neural networks.


# Installation

To install the latest version of this library run the following using ``pip``

```bash
$ pip install git+https://github.com/clementchadebec/benchmark_VAE.git
``` 


or alternatively you can clone the github repo to access to tests, tutorials and scripts.
```bash
$ git clone https://github.com/clementchadebec/pythae.git
```
and install the library
```bash
$ cd benchmark_VAE
$ pip install .
``` 

## Available Models

Below is the list of the models currently implemented in the library.


|               Models               |                                                                                    Training example                                                                                    |                     Paper                    |                           Official Implementation                          |
|:----------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------:|:--------------------------------------------------------------------------:|
| Autoencoder (AE)                   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/clementchadebec/benchmark_VAE/blob/main/examples/notebooks/models_training/ae_training.ipynb) |                                              |                                                                            |
| Variational Autoencoder (VAE)      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/clementchadebec/benchmark_VAE/blob/main/examples/notebooks/models_training/vae_training.ipynb) | [link](https://arxiv.org/pdf/1312.6114.pdf)  |
| Beta Variational Autoencoder (Beta_VAE) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/clementchadebec/benchmark_VAE/blob/main/examples/notebooks/models_training/beta_vae_training.ipynb) | [link](https://openreview.net/pdf?id=Sy2fzU9gl)  |                                                                            |
| Importance Weighted Autoencoder (IWAE) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/clementchadebec/benchmark_VAE/blob/main/examples/notebooks/models_training/iwae_training.ipynb) | [link](https://arxiv.org/pdf/1509.00519v4.pdf)  | [link](https://github.com/yburda/iwae)                                                                            |
| Wasserstein Autoencoder (WAE)      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/clementchadebec/benchmark_VAE/blob/main/examples/notebooks/models_training/wae_training.ipynb) | [link](https://arxiv.org/pdf/1711.01558.pdf) | [link](https://github.com/tolstikhin/wae)                                  |
| Info Variational Autoencoder (INFOVAE_MMD)      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/clementchadebec/benchmark_VAE/blob/main/examples/notebooks/models_training/info_vae_training.ipynb) | [link](https://arxiv.org/pdf/1706.02262.pdf) |                                   |
| VAMP Autoencoder (VAMP)            | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/clementchadebec/benchmark_VAE/blob/main/examples/notebooks/models_training/vamp_training.ipynb) | [link](https://arxiv.org/pdf/1705.07120.pdf) | [link](https://github.com/jmtomczak/vae_vampprior)                         |
| Hamiltonian VAE (HVAE)             | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/clementchadebec/benchmark_VAE/blob/main/examples/notebooks/models_training/hvae_training.ipynb) | [link](https://arxiv.org/pdf/1805.11328.pdf) | [link](https://github.com/anthonycaterini/hvae-nips)                       |
| Regularized AE with L2 decoder param (RAE_L2) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/clementchadebec/benchmark_VAE/blob/main/examples/notebooks/models_training/rae_l2_training.ipynb) | [link](https://arxiv.org/pdf/1903.12436.pdf) | [link](https://github.com/ParthaEth/Regularized_autoencoders-RAE-/tree/master/models/rae) |
| Regularized AE with gradient penalty (RAE_GP) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/clementchadebec/benchmark_VAE/blob/main/examples/notebooks/models_training/rae_gp_training.ipynb) | [link](https://arxiv.org/pdf/1903.12436.pdf) | [link](https://github.com/ParthaEth/Regularized_autoencoders-RAE-/tree/master/models/rae) |
| Riemannian Hamiltonian VAE (RHVAE) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/clementchadebec/benchmark_VAE/blob/main/examples/notebooks/models_training/rhvae_training.ipynb) | [link](https://arxiv.org/pdf/2105.00026.pdf) | |

**See [results](#Results) for all aforementionned models**

## Available Samplers

Below is the list of the models currently implemented in the library.

|                Samplers               |   Models  		  | Paper 											  | Official Implementation 				  |
|:-------------------------------------:|:-------------------:|:-------------------------------------------------:|:-----------------------------------------:|
| Normal prior (NormalSampler)                         | all models		  | [link](https://arxiv.org/pdf/1312.6114.pdf)		  |
| Gaussian mixture (GaussianMixtureSampler) | all models		  | [link](https://arxiv.org/pdf/1903.12436.pdf) 	  | [link](https://github.com/ParthaEth/Regularized_autoencoders-RAE-/tree/master/models/rae) |
| VAMP prior sampler (VAMPSampler)                   |    VAMP   		  | [link](https://arxiv.org/pdf/1705.07120.pdf) 	  | [link](https://github.com/jmtomczak/vae_vampprior) |
| Manifold sampler (RHVAESampler)                     |    RHVAE  		  | [link](https://arxiv.org/pdf/2105.00026.pdf)      |													|
| Two stage VAE sampler (TwoStageVAESampler)					| all VAE based models| [link](https://openreview.net/pdf?id=B1e0X3C9tQ)  | [link](https://github.com/daib13/TwoStageVAE/) |)


## Launching a model training

To launch a model training, you only need to call a `TrainingPipeline` instance. 

```python
>>> from pythae.pipelines import TrainingPipeline
>>> from pythae.models import VAE, VAEConfig
>>> from pythae.trainers import BaseTrainingConfig

>>> # Set up the training configuration
>>> my_training_config = BaseTrainingConfig(
...	output_dir='my_model',
...	num_epochs=50,
...	learning_rate=1e-3,
...	batch_size=200,
...	steps_saving=None
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
...	)
>>> # Launch the Pipeline
>>> pipeline(
...	train_data=your_train_data, # must be torch.Tensor or np.array 
...	eval_data=your_eval_data # must be torch.Tensor or np.array
...	)
```

At the end of training, the best model weights, model configuration and training configuration are stored in a `final_model` folder available in  `my_model/MODEL_NAME_training_YYYY-MM-DD_hh-mm-ss` (with `my_model` being the `output_dir` argument of the `TrainingConfig`). If you further set the `steps_saving` argument to a a certain value, folders named `checkpoint_epoch_k` containing the best model weights, configuration and training configuration at epoch *k* will also appear in `my_model/MODEL_NAME_training_YYYY-MM-DD_hh-mm-ss`.

## Lauching a training on benchmark datasets
We also provide a training script example [here](https://github.com/clementchadebec/benchmark_VAE/tree/main/examples/scripts/training.py) that can be used to train the models on benchmarks datasets (mnist, cifar10, celeba ...). The script can be launched with the following commandline

```bash
python training.py --dataset mnist --model_name ae --model_config 'configs/ae_config.json' --training_config 'configs/base_training_config.json'
```

See [README.md](https://github.com/clementchadebec/benchmark_VAE/tree/main/examples/scripts/README.md) for further details on this script

## Launching data generation

To launch the data generation process from a trained model, you only need to build you sampler and retrieve your 
Several samplers are available for each models please check here to see which ones apply to your vae.
, run the following.

```python
>>> from pythae.models import VAE
>>> from pythae.samplers import NormalSampler
>>> # Retrieve the trained model
>>> my_trained_vae = VAE.load_from_folder(
...	'path/to/your/trained/model'
...	)
>>> # Define your sampler
>>> my_samper = NormalSampler(
...	model=my_trained_vae
...	)
>>> # Generate samples
>>> gen_data = normal_samper.sample(
...	num_samples=50,
...	batch_size=10,
...	output_dir=None,
...	return_gen=True
...	)
```
If you set `output_dir` to a specific path the generated images will be saved as `.png` files named `00000000.png`, `00000001.png` ...
The samplers can be used with any model as long as it is suited. For instance, a `GMMSampler` instance can be used to generate from any model but a `VAMPSampler` will only be usable with a `VAMP` model


## Define you own Autoencoder architecture
 
Say you want to train a Wassertstein AE with a specific encoder and decoder. Pythae provides you the possibility to define your own neural networks as follows

```python
>>> from pythae.models.nn import BaseEncoder, BaseDecoder
>>> from pythae.models.base.base_utils import ModelOuput
>>> class My_Encoder(BaseEncoder):
...	def __init__(self, args=None): # Args is a ModelConfig instance
...		BaseEncoder.__init__(self)
...		self.layers = my_nn_layers()
...		
...	def forward(self, x:torch.Tensor) -> ModelOuput:
...		out = self.layers(x)
...		output = ModelOuput(
...			embedding=out # Set the output from the encoder in a ModelOuput instance 
...		)
...		return output
...
... class My_Decoder(BaseDecoder):
...	def __init__(self, args=None):
...		BaseDecoder.__init__(self)
...		self.layers = my_nn_layers()
...		
...	def forward(self, x:torch.Tensor) -> ModelOuput:
...		out = self.layers(x)
...		output = ModelOuput(
...			reconstruction=out # Set the output from the decoder in a ModelOuput instance
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

## Using benchmark neural nets
You may also find predefined neural network architecture for the most common data sets (*i.e.* MNIST, CIFAR, CELEBA ...) that can be loaded as follows

```python
>>> for pythae.models.nn.benchmark.mnist import (
...	Encoder_AE_MNIST, # For AE based model (only return embeddings)
... 	Encoder_VAE_MNIST, # For VAE based model (return embeddings and log_covariances)
... 	Decoder_AE_MNIST
)
```
Replace *mnist* by cifar or celeba to access to other neural nets.

## Getting your hands on the code

To help you to understand the way pythae works and how you can augment your data with this library we also
provide tutorials that can be found in [examples folder](https://github.com/clementchadebec/benchmark_VAE/tree/main/examples):

- [making_your_own_autoencoder.ipynb](https://github.com/clementchadebec/benchmark_VAE/tree/main/examples/notebooks) shows you how to pass your own networks to the models implemented in pythae [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/clementchadebec/benchmark_VAE/blob/main/examples/notebooks/making_your_own_autoencoder.ipynb)

- [models_training](https://github.com/clementchadebec/benchmark_VAE/tree/main/examples/notebooks/models_training) folder provides notebooks showing how to train each implemented models and how to sample from them using `pyhtae.samplers`.

- [scripts](https://github.com/clementchadebec/benchmark_VAE/tree/main/examples/scripts) folder provides in particular an example of a training script to train the models on benchmark data sets (mnist, cifar10, celeba ...)

## Dealing with issues

If you are experiencing any issues while running the code or request new features/models to be implemented please [open an issue on github](https://github.com/clementchadebec/benchmark_VAE/issues).


## Results
(WIP)

|               Models               |                                                                                    MNIST                                                                     |                     CELEBA             
|:----------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------:|
| AE  + GaussianMixtureSampler                  | ![AE GMM](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/ae_gmm_sampling_mnist.png) | ![AE GMM](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/ae_gmm_sampling_celeba.png)                                                                            |
| VAE  + NormalSampler    | ![VAE Normal](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/vae_normal_sampling_mnist.png) |  ![VAE Normal](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/vae_normal_sampling_celeba.png)
| Beta-VAE + NormalSampler | ![Beta Normal](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/beta_vae_normal_sampling_mnist.png) | ![Beta Normal](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/beta_vae_normal_sampling_celeba.png)
| IWAE +  Normal sampler | ![IWAE Normal](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/iwae_normal_sampling_mnist.png) | ![IWAE Normal](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/iwae_normal_sampling_celeba.png)
| WAE + NormalSampler| ![WAE Normal](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/wae_normal_sampling_mnist.png) | ![WAE Normal](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/wae_normal_sampling_celeba.png)
| INFO VAE + NormalSampler| ![INFO Normal](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/infovae_normal_sampling_mnist.png) | ![INFO Normal](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/infovae_normal_sampling_celeba.png)
| VAMP + VAMPSampler          | ![VAMP Vamp](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/vamp_vamp_sampling_mnist.png) | ![VAMP Vamp](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/vamp_vamp_sampling_celeba.png) |
| HVAE + GaussianMixtureSampler             | ![HVAE GMM](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/hvae_gmm_sampling_mnist.png) | ![HVAE GMM](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/hvae_gmm_sampling_celeba.png)
| RAE_L2 + GaussianMixtureSampler | ![RAE L2 GMM](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/rae_l2_gmm_sampling_mnist.png)  |  ![RAE L2 GMM](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/rae_l2_gmm_sampling_celeba.png)
| RAE_GP + GaussianMixtureSampler| ![RAE GMM](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/rae_gp_gmm_sampling_mnist.png)  |  ![RAE GMM](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/rae_gp_gmm_sampling_celeba.png)
| Riemannian Hamiltonian VAE (RHVAE) + RHVAE Sampler| ![RHVAE RHVAE](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/rhvae_rhvae_sampling_mnist.png) | ![RHVAE RHVAE](https://github.com/clementchadebec/benchmark_VAE/blob/main/examples/showcases/rhvae_rhvae_sampling_celeba.png)
