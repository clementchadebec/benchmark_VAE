## Scripts

We also provided a training script as example allowing you to train VAE models on well known benchmark data set (mnist, cifar10, celeba ...).
The script can be launched with the following commandline

```bash
python training.py --dataset mnist --model_name ae --model_config 'configs/ae_config.json' --training_config 'configs/base_training_config.json'
```

The folder structure should be as follows:
```bash
.
├── configs # the model & training config files (you can amend these files as desired or specify the location of yours in '--model_config' )
│   ├── ae_config.json
│   ├── base_training_config.json
│   ├── beta_vae_config.json
│   ├── hvae_config.json
│   ├── rhvae_config.json
│   ├── vae_config.json
│   └── vamp_config.json
├── data # the dataset with train_data.npz and eval_data.npz files
│   ├── celeba
│   │   ├── eval_data.npz
│   │   └── train_data.npz
│   ├── cifar10
│   │   ├── eval_data.npz
│   │   └── train_data.npz
│   └── mnist
│       ├── eval_data.npz
│       └── train_data.npz
├── my_models # trained models are saved here
│   ├── AE_training_2021-10-15_16-07-04 
│   └── RHVAE_training_2021-10-15_15-54-27
├── README.md
└── training.py
```

**Note** The data in the `train_data.npz` and `eval_data.npz` files must be loadable as follows

```python
train_data = np.load(os.path.join(PATH, f'data/{args.dataset}', 'train_data.npz'))['data']
eval_data = np.load(os.path.join(PATH, f'data/{args.dataset}', 'eval_data.npz'))['data']
```
where `train_data` and `eval_data` have now the shape (n_img x im_channel x height x width)