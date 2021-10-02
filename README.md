
<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/clementchadebec/pyraug/main/docs/source/imgs/logo_pyraug_2.jpeg" width="400"/>
    <br>
<p>

<p align="center">
<a href='https://pypi.org/project/pyraug/'>
	    <img src='https://badge.fury.io/py/pyraug.svg' />
	</a>
	<a href='https://opensource.org/licenses/Apache-2.0'>
	    <img src='https://img.shields.io/github/license/clementchadebec/pyraug?color=blue' />
	</a>
	<a href='https://pyraug.readthedocs.io/en/latest/?badge=latest'>
	    <img src='https://readthedocs.org/projects/pyraug/badge/?version=latest' alt='Documentation 	Status' />
	</a>
	<a href='https://pepy.tech/project/pyraug'>
	    <img src='https://static.pepy.tech/personalized-badge/pyraug?period=month&units=international_system&left_color=grey&right_color=orange&left_text=downloads' alt='Downloads 	Status' />
	</a>
</p>
<p align="center">
  <a href="https://pyraug.readthedocs.io/en/latest/">Documentation</a>
</p>
	


# Pyraug 

This library provides a way to perform Data Augmentation using Variational Autoencoders in a 
reliable way even in challenging contexts such as high dimensional and low sample size 
data.


# Installation

To install the library from [pypi.org](https://pypi.org/) run the following using ``pip``

```bash
$ pip install pyraug
``` 


or alternatively you can clone the github repo to access to tests, tutorials and scripts.
```bash
$ git clone https://github.com/clementchadebec/pyraug.git
```
and install the library
```bash
$ cd pyraug
$ pip install .
``` 

# Augmenting your Data


In Pyraug, a typical augmentation process is divided into 2 distinct parts:

1. Train a model using the Pyraug's ```TrainingPipeline``` or using the provided ``scripts/training.py`` script
2. Generate new data from a trained model using Pyraug's ```GenerationPipeline``` or using the provided ``scripts/generation.py`` script

There exist two ways to augment your data pretty straightforwardly using Pyraug's built-in functions. 


## Using Pyraug's Pipelines

Pyraug provides two pipelines that may be used to either train a model on your own data or generate new data with a pretrained model.


**note**: These pipelines are independent of the choice of the model and sampler. Hence, they can be used even if you want to access to more advanced features such as defining your own autoencoding architecture. 

### Launching a model training


To launch a model training, you only need to call a `TrainingPipeline` instance. 
In its most basic version the `TrainingPipeline` can be built without any arguments.
This will by default train a `RHVAE` model with default autoencoding architecture and parameters.

```python
>>> from pyraug.pipelines import TrainingPipeline
>>> pipeline = TrainingPipeline()
>>> pipeline(train_data=dataset_to_augment)
```

where ``dataset_to_augment`` is either a `numpy.ndarray`, `torch.Tensor` or a path to a folder where each file is a data (handled data format are ``.pt``, ``.nii``, ``.nii.gz``, ``.bmp``, ``.jpg``, ``.jpeg``, ``.png``). 

More generally, you can instantiate your own model and train it with the `TrainingPipeline`. For instance, if you want to instantiate a basic `RHVAE` run:


```python
>>> from pyraug.models import RHVAE
>>> from pyraug.models.rhvae import RHVAEConfig
>>> model_config = RHVAEConfig(
...    input_dim=int(intput_dim)
... ) # input_dim is the shape of a flatten input data
...   # needed if you do not provided your own architectures
>>> model = RHVAE(model_config)
```


In case you instantiate yourself a model as shown above and you do not provided all the network architectures (encoder, decoder & metric if applicable), the `ModelConfig` instance will expect you to provide the input dimension of your data which equals to ``n_channels x height x width x ...``. Pyraug's VAE models' networks indeed default to Multi Layer Perceptron neural networks which automatically adapt to the input data shape. 

**note**: In case you have different size of data, Pyraug will reshape it to the minimum size ``min_n_channels x min_height x min_width x ...``



Then the `TrainingPipeline` can be launched by running:

```python
>>> from pyraug.pipelines import TrainingPipeline
>>> pipe = TrainingPipeline(model=model)
>>> pipe(train_data=dataset_to_augment)
```

At the end of training, the model weights ``models.pt`` and model config ``model_config.json`` file 
will be saved in a folder ``outputs/my_model/training_YYYY-MM-DD_hh-mm-ss/final_model``. 

**Important**: For high dimensional data we advice you to provide you own network architectures and potentially adapt the training and model parameters see [documentation](https://pyraug.readthedocs.io/en/latest/advanced_use.html) for more details.


### Launching data generation


To launch the data generation process from a trained model, run the following.

```python
>>> from pyraug.pipelines import GenerationPipeline
>>> from pyraug.models import RHVAE
>>> model = RHVAE.load_from_folder('path/to/your/trained/model') # reload the model
>>> pipe = GenerationPipeline(model=model) # define pipeline
>>> pipe(samples_number=10) # This will generate 10 data points
```

The generated data is in ``.pt`` files in ``dummy_output_dir/generation_YYYY-MM-DD_hh-mm-ss``. By default, it stores batch data of a maximum of 500 samples.



### Retrieve generated data

Generated data can then be loaded pretty easily by running

```python
>>> import torch
>>> data = torch.load('path/to/generated_data.pt')

```

## Using the provided scripts


Pyraug provides two scripts allowing you to augment your data directly with commandlines.


**note**: To access to the predefined scripts you should first clone the Pyraug's repository.
The following scripts are located in [scripts folder](https://github.com/clementchadebec/pyraug/tree/main/scripts). For the time being, only `RHVAE` model training and generation is handled by the provided scripts. Models will be added as they are implemented in [pyraug.models](https://github.com/clementchadebec/pyraug/tree/main/src/pyraug/models) 


### Launching a model training:

To launch a model training, run 

```
$ python scripts/training.py --path_to_train_data "path/to/your/data/folder" 
```


The data must be located in ``path/to/your/data/folder`` where each input data is a file. Handled image types are ``.pt``, ``.nii``, ``.nii.gz``, ``.bmp``, ``.jpg``, ``.jpeg``, ``.png``. Depending on the usage, other types will be progressively added.


At the end of training, the model weights ``models.pt`` and model config ``model_config.json`` file 
will be saved in a folder ``outputs/my_model_from_script/training_YYYY-MM-DD_hh-mm-ss/final_model``. 


### Launching data generation


Then, to launch the data generation process from a trained model, you only need to run 

```
$ python scripts/generation.py --num_samples 10 --path_to_model_folder 'path/to/your/trained/model/folder' 
```


The generated data is stored in several ``.pt`` files in ``outputs/my_generated_data_from_script/generation_YYYY-MM-DD_hh_mm_ss``. By default, it stores batch data of 500 samples.



**Important**:  In the simplest configuration, default configurations are used in the scripts. You can easily override as explained in [documentation](https://pyraug.readthedocs.io/en/latest/advanced/setting_configs.html). See tutorials for a more in depth example.



### Retrieve generated data

Generated data can then be loaded pretty easily by running

```python
>>> import torch
>>> data = torch.load('path/to/generated_data.pt')
```



## Getting your hands on the code

To help you to understand the way Pyraug works and how you can augment your data with this library we also
provide tutorials that can be found in [examples folder](https://github.com/clementchadebec/pyraug/tree/main/examples):

- [getting_started.ipynb](https://github.com/clementchadebec/pyraug/tree/main/examples) explains you how to train a model and generate new data using Pyraug's Pipelines [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/clementchadebec/pyraug/blob/main/examples/getting_started.ipynb)
- [playing_with_configs.ipynb](https://github.com/clementchadebec/pyraug/tree/main/examples) shows you how to amend the predefined configuration to adapt them to you data [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/clementchadebec/pyraug/blob/main/examples/playing_with_configs.ipynb)
- [making_your_own_autoencoder.ipynb](https://github.com/clementchadebec/pyraug/tree/main/examples) shows you how to pass your own networks to the models implemented in Pyraug [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/clementchadebec/pyraug/blob/main/examples/making_your_own_autoencoder.ipynb)

## Dealing with issues

If you are experiencing any issues while running the code or request new features please [open an issue on github](https://github.com/clementchadebec/pyraug/issues)


## Citing

If you use this library please consider citing us:

```bibtex
@article{chadebec_data_2021,
	title = {Data {Augmentation} in {High} {Dimensional} {Low} {Sample} {Size} {Setting} {Using} a {Geometry}-{Based} {Variational} {Autoencoder}},
	copyright = {All rights reserved},
	journal = {arXiv preprint arXiv:2105.00026},
  	arxiv = {2105.00026},
	author = {Chadebec, Clément and Thibeau-Sutre, Elina and Burgos, Ninon and Allassonnière, Stéphanie},
	year = {2021}
}
```
```bibtex
@article{chadebec_geometry-aware_2020,
	title = {Geometry-{Aware} {Hamiltonian} {Variational} {Auto}-{Encoder}},
	copyright = {All rights reserved},
	journal = {arXiv preprint arXiv:2010.11518},
    	arxiv = {2010.11518},
	author = {Chadebec, Clément and Mantoux, Clément and Allassonnière, Stéphanie},
	year = {2020}
}


```

### Credits
Logo: [SaulLu](https://github.com/saullu)
