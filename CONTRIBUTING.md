# Welcome to the contributing guidelines !
Want to contribute to pythae library ? That is cool! Thank you! :smile:

## Contributing guidelines

If you want to contribute to this repo, please consider following this checklist

1) Fork this repo
2) Clone it on your local drive and install the library in editable mode
```bash
$ git clone git@github.com:your_github_name/benchmark_VAE.git
$ cd benchmark_VAE
$ pip install -e .
```
3) Create a branch with an explicit name of your contribution
```bash
$ git checkout -b my_branch_with_contribution
```
4) Make sure you add the appropriate tests to test your feature. If the library test coverage reduces
significantly, the contribution will raise some red flags.

5) Ensure that your contribution passes the existing test suite by running
```bash
pytest tests/
``` 
- Polish your contribution using black and isort

- Finally, open a pull request directly on your Github ! :rocket: 


## Implementing a new model
If you want to add a new model please make sure that you followed the following checklist:
- [ ] Create a folder named `your_model_name` in `pythae/models` and containg a file with the model implementation entitled `your_model_name_model.py` and a file with the model configuration named `your_model_name_config.py`.
- [ ] The `your_model_name_model.py` file contains a class with the name of your model inheriting from either 
    `AE` or `VAE` classes depending on the model architecture. 
- [ ] The `your_model_name_config.py` files contains a dataclass inheriting from either `AEConfig` or `VAEConfig`. See for instance `pythae/models/rae_l2` folder for a AE-based models and `pythae/models/rhvae` folder for a VAE-based models
- [ ] The model must have a forward method in which the loss is computed and returning a `ModelOutput` instance with the loss being stored under the`loss` key.
- [ ] You also implemented the classmethods `load_from_folder` and `_load_model_config_from_folder` allowing to reload the model from a folder. See `pythae/models/rae_l2` for instance.
- [ ] Make your tests in the `tests` folder. See for instance `pythae/tests/test_rae_l2_tests.py`. You will see that the tests for the models look the same and cover them quite well. Hence, you can reuse this test suite as an inspiration to test your model.

## Implementing a new sampler
If you want to add a new sampler please make sure that you followed the following checklist:
- [ ] Create a folder named `your_sampler_name` in `pythae/samplers` and containg a file with the sampler implementation entitled `your_sampler_name_sampler.py` and a file with the sampler configuration (if needed) named `your_sampler_name_config.py`. See `pythae/samplers/gaussian_mixture` for instance.
- [ ] The `your_sampler_name_sampler.py` files contains a class with the name of yoyr sampler inheriting from `BaseSampler`.
- [ ] The `your_sampler_name_config.py` files contains a dataclass inheriting from `BaseSamplerConfig`. See `pythae/samplers/gaussian_mixture/gaussian_mixture_config.py`.
- [ ] The sampler must have a `sample` method able to save the generated images in a folder and return them if desired.
- [ ] If the sampler needs to be fitted, a `fit` merhod can be implemented. See `pythae/samplers/gaussian_mixture/gaussian_mixture_samplers.py` for example.
- [ ] Make your tests in the `tests` folder. See for instance `pythae/tests/test_gaussian_mixture_sampler.py`. You will see that the tests for the samplers look the same, you can reuse this test suite as an inspiration to test your sampler.

## Any doubts ?
In any case if you have any question, do not hesitate to reach out to me directly, I will be happy to help! :smile:
