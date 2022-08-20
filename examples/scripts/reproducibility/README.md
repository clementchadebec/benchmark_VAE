## Reproducibility

We validate the implementations by reproducing some results presented in the original publications when the official code has been released or when enough details about the experimental section of the papers were available (we indeed noted that in many papers key elements for reproducibility were missing such as the data split considered, which criteria is used to select the model on which the metrics are computed, the hyper-parameters are not fully disclosed or the network architectures is unclear making reproduction very hard if not impossible in certain cases). If you succeed in reproducing a result that is not listed below, I would be very happy to had it citing you! :)

We gather here the script to train those models. They can be launched with the following commandline

```bash
python aae.py --model_config 'configs/celeba/aae/aae_config.json' --training_config 'configs/celeba/aae/base_training_config.json'
```

The folder structure should be as follows:
```bash
.
├── aae.py
├── betatcvae.py
├── configs
│   ├── binary_mnist
│   ├── celeba
│   ├── dsprites
│   └── mnist
├── data
│   ├── binary_mnist
│   ├── celeba
│   ├── cifar10
│   ├── dsprites
│   ├── fashion
│   └── mnist
├── factorvae.py
├── hvae.py
├── iwae.py
├── rae_gp.py
├── rae_l2.py
├── svae.py
├── vae_nf.py
├── vae.py
├── vamp.py
└── wae.py
```

**Note** The data in the `data folder/dataset_name` must have `train_data.npz` and `eval_data.npz` files that must be loadable as follows

```python
train_data = np.load(os.path.join(PATH, f'data/{args.dataset}', 'train_data.npz'))['data']
eval_data = np.load(os.path.join(PATH, f'data/{args.dataset}', 'eval_data.npz'))['data']
```
where `train_data` and `eval_data` have now the shape (n_img x im_channel x height x width)

Below are gathered the results we were able to reproduce