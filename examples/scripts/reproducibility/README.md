## Reproducibility

We validate the implementations by reproducing some results presented in the original publications when the official code has been released or when enough details about the experimental section of the papers were available (we indeed noted that in many papers key elements for reproducibility were missing such as the data split considered, which criteria is used to select the model on which the metrics are computed, the hyper-parameters are not fully disclosed or the network architectures is unclear making reproduction very hard if not impossible in certain cases). If you succeed in reproducing a result that is not listed below, I would be very happy to add it citing you! :)

We gather here the scripts to train those models. They can be launched with the following commandline

```bash
python aae.py
```

The folder structure should be as follows:
```bash
.
├── aae.py
├── betatcvae.py
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

| Model | Dataset | Metric | Obtained value | Reference value| Reference (paper/code) | Trained model
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| VAE | Binary MNIST | NLL (200 IS) | 89.78 (0.01) | 89.9 | [paper](https://arxiv.org/abs/1505.05770) | [link](https://huggingface.co/clementchadebec/reproduced_vae)
| VAMP (K=500) | Binary MNIST | NLL (5000 IS) | 85.79 (0.00) | 85.57 | [paper](https://arxiv.org/abs/1705.07120) | [link](https://huggingface.co/clementchadebec/reproduced_vamp)
| SVAE | Dyn. Binarized MNIST | NLL (500 IS) | 93.13 (0.01) | 93.16 (0.31) | [code](https://github.com/nicola-decao/s-vae-pytorch) | [link](https://huggingface.co/clementchadebec/reproduced_svae) |
PoincareVAE (Wrapped)| MNIST | NLL (500 IS) | 101.66 (0.00) | 101.47 (0.01) | [code](https://github.com/emilemathieu/pvae) | [link](https://huggingface.co/clementchadebec/reproduced_wrapped_poincare_vae)
| IWAE (n_samples=50) | Binary MNIST | NLL (5000 IS) | 86.82 (0.01) | 87.1 | [paper](https://arxiv.org/abs/1509.00519) | [link](https://huggingface.co/clementchadebec/reproduced_iwae)
| MIWAE (M=8, K=8) | Dyn. Binarized MNIST | NLL (5000 IS) | 85.09 (0.00) | 84.97 (0.10) | [paper](https://arxiv.org/abs/1802.04537) | [link](https://huggingface.co/clementchadebec/reproduced_miwae)
| PIWAE (M=8, K=8) | Dyn. Binarized MNIST | NLL (5000 IS) | 84.58 (0.00) | 84.46 (0.06) | [paper](https://arxiv.org/abs/1802.04537) | [link](https://huggingface.co/clementchadebec/reproduced_piwae)
| CIWAE (beta=0.05) | Dyn. Binarized MNIST | NLL (5000 IS) | 84.74 (0.01) | 84.57 (0.09) | [paper](https://arxiv.org/abs/1802.04537) | [link](https://huggingface.co/clementchadebec/reproduced_ciwae)
| HVAE (n_lf=4) | Binary MNIST | NLL (1000 IS) | 86.21 (0.01) | 86.40 | [paper](https://arxiv.org/abs/1410.6460) | [link](https://huggingface.co/clementchadebec/reproduced_hvae)
| BetaTCVAE | DSPRITES | Modified ELBO/ELBO (after 50 epochs) | 710.41/85.54 | 712.26/86.40 | [code](https://github.com/rtqichen/beta-tcvae) | [link](https://huggingface.co/clementchadebec/reproduced_beta_tc_vae)
| RAE_L2 | MNIST | FID | 9.1 | 9.9 | [code](https://github.com/ParthaEth/Regularized_autoencoders-RAE-) | [link](https://huggingface.co/clementchadebec/reproduced_rae_l2)
| RAE_GP | MNIST | FID | 9.7 | 9.4 | [code](https://github.com/ParthaEth/Regularized_autoencoders-RAE-)| [link](https://huggingface.co/clementchadebec/reproduced_rae_gp)
| WAE | CELEBA 64 | FID | 56.5 | 55 | [paper](https://arxiv.org/abs/1711.01558) | [link](https://huggingface.co/clementchadebec/reproduced_wae)
| AAE | CELEBA 64 | FID | 43.3 | 42 | [paper](https://arxiv.org/abs/1711.01558)| [link](https://huggingface.co/clementchadebec/reproduced_aae)

