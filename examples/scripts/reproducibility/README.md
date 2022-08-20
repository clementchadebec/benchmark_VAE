## Reproducibility

We validate the implementations by reproducing some results presented in the original publications when the official code has been released or when enough details about the experimental section of the papers were available (we indeed noted that in many papers key elements for reproducibility were missing such as the data split considered, which criteria is used to select the model on which the metrics are computed, the hyper-parameters are not fully disclosed or the network architectures is unclear making reproduction very hard if not impossible in certain cases). If you succeed in reproducing a result that is not listed below, I would be very happy to add it citing you! :)

We gather here the scripts to train those models. They can be launched with the following commandline

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
train_data = np.load(os.path.join(PATH, f'data/dataset', 'train_data.npz'))['data']
eval_data = np.load(os.path.join(PATH, f'data/dataset', 'eval_data.npz'))['data']
```
where `train_data` and `eval_data` have now the shape (n_img x im_channel x height x width)

Below are gathered the results we were able to reproduce

| Model | Dataset | Metric | Obtained value | Reference value |
|:---:|:---:|:---:|:---:|:---:|
| VAE | Binary MNIST | NLL (200 IS) | 89.78 (0.01) | 89.9 (0.31) |
| VAMP (K=500) | Binary MNIST | NLL (5000 IS) | 85.79 (0.00) | 85.57 |
| SVAE | Dyn. Binarized MNIST | NLL (500 IS) | 93.27 (0.69) | 93.16 (0.31) |
| WAE (n_samples=50) | Binary MNIST | NLL (5000 IS) | 86.82 (0.01) | 87.1 |
| HVAE (n_lf=4) | Binary MNIST | NLL (1000 IS) | 86.21 (0.01) | 86.40 |
| BetaTCVAE | DSPRITES | ELBO/Modified ELBO (after 50 epochs) | 710.41/85.54 | 712.26/86.40 |
| RAE_L2 | MNIST | FID | 9.1 | 9.9 |
| RAE_GP | MNIST | FID | 9.7 | 9.4 |
| WAE | CELEBA 64 | FID | 56.5 | 55 |
| AAE | CELEBA 64 | FID | 43.3 | 42 |