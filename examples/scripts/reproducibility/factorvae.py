import argparse
import logging
import os
from typing import List

import numpy as np
import torch

from pythae.data.preprocessors import DataProcessor
from pythae.models import AutoModel
from pythae.trainers import AdversarialTrainer, AdversarialTrainerConfig

from pythae.models.nn import BaseEncoder, BaseDecoder
import torch.nn as nn
from pythae.models.base.base_utils import ModelOutput


logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)

PATH = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Training setting
ap.add_argument(
    "--dataset",
    type=str,
    default="dsprites",
    choices=["mnist", "binary_mnist", "dsprites"],
    help="The data set to use to perform training. It must be located in the folder 'data' at the "
    "path 'data/datset_name/' and contain a 'train_data.npz' and a 'eval_data.npz' file with the "
    "data being under the key 'data'. The data must be in the range [0-255] and shaped with the "
    "channel in first position (im_channel x height x width).",
    required=True,
)
ap.add_argument(
    "--model_name",
    help="The name of the model to train",
    choices=[
        "vae",
        "beta_vae",
        "iwae",
        "wae",
        "info_vae",
        "vamp",
        "hvae",
        "rhvae",
        "aae",
        "vaegan",
        "vqvae",
        "msssim_vae",
        "svae",
        "factor_vae",
        "beta_tc_vae",
        "vae_nf",
        "vae_iaf",
        "vae_lin_nf",
    ],
    required=True,
)
ap.add_argument(
    "--model_config",
    help="path to model config file (expected json file)",
    default=None,
)
ap.add_argument(
    "--training_config",
    help="path to training config_file (expected json file)",
    default=os.path.join(PATH, "configs/base_training_config.json"),
)
ap.add_argument(
    "--use_wandb",
    help="whether to log the metrics in wandb",
    action="store_true",
)
ap.add_argument(
    "--wandb_project",
    help="wandb project name",
    default="nll",
)
ap.add_argument(
    "--wandb_entity",
    help="wandb entity name",
    default="benchmark_team",
)

args = ap.parse_args()

def unif_init(m, n_in, n_out):
    scale = np.sqrt(6./(n_in+n_out))
    m.weight.data.uniform_(
            -scale, scale
        )
    m.bias.data = torch.zeros((1, n_out))

class Encoder(BaseEncoder):
    def __init__(self, args: dict):
        super(Encoder, self).__init__()
        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim


        self.layers = nn.Sequential(
                nn.Conv2d(1, 32, 4, 2),
                nn.ReLU(),
                nn.Conv2d(32, 32, 4, 2),
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, 2),
                nn.ReLU(),
                nn.Conv2d(64, 64, 4, 2),
                nn.ReLU()
            )

        self.fc = nn.Linear(256, 128)

        self.embedding = nn.Linear(128, self.latent_dim)
        self.log_var = nn.Linear(128, self.latent_dim)


    def forward(self, x):
        output = ModelOutput()

        out = self.layers(x)
        out = self.fc(out.reshape(-1, 256))

        output["embedding"] = self.embedding(out)
        output["log_covariance"] = self.log_var(out)

        return output


class Decoder(BaseDecoder):
    def __init__(self, args: dict):
        BaseDecoder.__init__(self)

        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim

        self.fc = nn.Sequential(
            nn.Linear(self.latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1024),
            nn.ReLU()
        )

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 4, 2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, padding=1),
            nn.Sigmoid()
        )


    def forward(self, z: torch.Tensor, output_layer_levels: List[int] = None):

        output = ModelOutput()

        out = self.fc(z)

        out = self.layers(out.reshape(-1, 64, 4, 4))
        output["reconstruction"] = out.reshape((z.shape[0],) + self.input_dim)

        return output



def main(args):

    
    try:
        if args.dataset == "mnist":
            logger.info(f"\nLoading {args.dataset} data...\n")
            train_data = (
                np.load(os.path.join(PATH, f"data/{args.dataset}", "train_data.npz"))[
                    "data"
                ]
                / 255.0
            )
            eval_data = (
                np.load(os.path.join(PATH, f"data/{args.dataset}", "eval_data.npz"))["data"]
                / 255.0
            )
            test_data = (
                np.load(os.path.join(PATH, f"data/{args.dataset}", "test_data.npz"))["data"]
                / 255.0
            )

        elif args.dataset == "dsprites":
            data = np.load(os.path.join(PATH, f"data/{args.dataset}", "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"), encoding='latin1')
            
            idx = torch.randperm(100000)
            
            train_data = torch.from_numpy(data['imgs']).float().reshape(-1, 1, 64, 64)[idx[:80000]]
            eval_data = torch.from_numpy(data['imgs']).float().reshape(-1, 1, 64, 64)[idx[80000:90000]]
            test_data = torch.from_numpy(data['imgs']).float().reshape(-1, 1, 64, 64)[idx[90000:100000]]

        else:
            logger.info(f"\nLoading {args.dataset} data...\n")
            train_data = np.loadtxt(os.path.join(PATH, f"data/{args.dataset}", "binarized_mnist_train.amat"))
            eval_data = np.loadtxt(os.path.join(PATH, f"data/{args.dataset}", "binarized_mnist_valid.amat"))
            test_data = np.loadtxt(os.path.join(PATH, f"data/{args.dataset}", "binarized_mnist_test.amat"))


    except Exception as e:
        raise FileNotFoundError(
            f"Unable to load the data from 'data/{args.dataset}' folder. Please check that both a "
            "'train_data.npz' and 'eval_data.npz' are present in the folder.\n Data must be "
            " under the key 'data', in the range [0-255] and shaped with channel in first "
            "position\n"
            f"Exception raised: {type(e)} with message: " + str(e)
        ) from e

    logger.info("Successfully loaded data !\n")
    logger.info("------------------------------------------------------------")
    logger.info("Dataset \t \t Shape \t \t \t Range")
    logger.info(
        f"{args.dataset.upper()} train data: \t {train_data.shape} \t [{train_data.min()}-{train_data.max()}] "
    )
    logger.info(
        f"{args.dataset.upper()} eval data: \t {eval_data.shape} \t [{eval_data.min()}-{eval_data.max()}]"
    )
    logger.info("------------------------------------------------------------\n")

    data_input_dim = tuple(train_data.shape[1:])


    if args.model_name == "vae":
        from pythae.models import VAE, VAEConfig

        if args.model_config is not None:
            model_config = VAEConfig.from_json_file(args.model_config)

        else:
            model_config = VAEConfig()

        model_config.input_dim = data_input_dim

        model = VAE(
            model_config=model_config,
            encoder=Encoder(model_config),
            decoder=Decoder(model_config),
        )

    elif args.model_name == "iwae":
        from pythae.models import IWAE, IWAEConfig

        if args.model_config is not None:
            model_config = IWAEConfig.from_json_file(args.model_config)

        else:
            model_config = IWAEConfig()

        model_config.input_dim = data_input_dim

        model = IWAE(
            model_config=model_config,
            encoder=Encoder(model_config),
            decoder=Decoder(model_config),
        )

    elif args.model_name == "info_vae":
        from pythae.models import INFOVAE_MMD, INFOVAE_MMD_Config

        if args.model_config is not None:
            model_config = INFOVAE_MMD_Config.from_json_file(args.model_config)

        else:
            model_config = INFOVAE_MMD_Config()

        model_config.input_dim = data_input_dim

        model = INFOVAE_MMD(
            model_config=model_config,
            encoder=Encoder(model_config),
            decoder=Decoder(model_config),
        )

    elif args.model_name == "wae":
        from pythae.models import WAE_MMD, WAE_MMD_Config

        if args.model_config is not None:
            model_config = WAE_MMD_Config.from_json_file(args.model_config)

        else:
            model_config = WAE_MMD_Config()

        model_config.input_dim = data_input_dim

        model = WAE_MMD(
            model_config=model_config,
            encoder=Encoder(model_config),
            decoder=Decoder(model_config),
        )

    elif args.model_name == "vamp":
        from pythae.models import VAMP, VAMPConfig

        if args.model_config is not None:
            model_config = VAMPConfig.from_json_file(args.model_config)

        else:
            model_config = VAMPConfig()

        model_config.input_dim = data_input_dim

        model = VAMP(
            model_config=model_config,
            encoder=Encoder(model_config),
            decoder=Decoder(model_config),
        )

    elif args.model_name == "beta_vae":
        from pythae.models import BetaVAE, BetaVAEConfig

        if args.model_config is not None:
            model_config = BetaVAEConfig.from_json_file(args.model_config)

        else:
            model_config = BetaVAEConfig()

        model_config.input_dim = data_input_dim

        model = BetaVAE(
            model_config=model_config,
            encoder=Encoder(model_config),
            decoder=Decoder(model_config),
        )

    elif args.model_name == "hvae":
        from pythae.models import HVAE, HVAEConfig

        if args.model_config is not None:
            model_config = HVAEConfig.from_json_file(args.model_config)

        else:
            model_config = HVAEConfig()

        model_config.input_dim = data_input_dim

        model = HVAE(
            model_config=model_config,
            encoder=Encoder(model_config),
            decoder=Decoder(model_config),
        )

    elif args.model_name == "rhvae":
        from pythae.models import RHVAE, RHVAEConfig

        if args.model_config is not None:
            model_config = RHVAEConfig.from_json_file(args.model_config)

        else:
            model_config = RHVAEConfig()

        model_config.input_dim = data_input_dim

        model = RHVAE(
            model_config=model_config,
            encoder=Encoder(model_config),
            decoder=Decoder(model_config),
        )

    elif args.model_name == "aae":
        from pythae.models import Adversarial_AE, Adversarial_AE_Config

        if args.model_config is not None:
            model_config = Adversarial_AE_Config.from_json_file(args.model_config)

        else:
            model_config = Adversarial_AE_Config()

        model_config.input_dim = data_input_dim

        model = Adversarial_AE(
            model_config=model_config,
            encoder=Encoder(model_config),
            decoder=Decoder(model_config),
        )


    elif args.model_name == "msssim_vae":
        from pythae.models import MSSSIM_VAE, MSSSIM_VAEConfig

        if args.model_config is not None:
            model_config = MSSSIM_VAEConfig.from_json_file(args.model_config)

        else:
            model_config = MSSSIM_VAEConfig()

        model_config.input_dim = data_input_dim

        model = MSSSIM_VAE(
            model_config=model_config,
            encoder=Encoder(model_config),
            decoder=Decoder(model_config),
        )

    elif args.model_name == "factor_vae":
        from pythae.models import FactorVAE, FactorVAEConfig

        if args.model_config is not None:
            model_config = FactorVAEConfig.from_json_file(args.model_config)

        else:
            model_config = FactorVAEConfig()

        model_config.input_dim = data_input_dim

        model = FactorVAE(
            model_config=model_config,
            encoder=Encoder(model_config),
            decoder=Decoder(model_config),
        )

    elif args.model_name == "beta_tc_vae":
        from pythae.models import BetaTCVAE, BetaTCVAEConfig

        if args.model_config is not None:
            model_config = BetaTCVAEConfig.from_json_file(args.model_config)

        else:
            model_config = BetaTCVAEConfig()

        model_config.input_dim = data_input_dim

        model = BetaTCVAE(
            model_config=model_config,
            encoder=Encoder(model_config),
            decoder=Decoder(model_config),
        )

    elif args.model_name == "vae_iaf":
        from pythae.models import VAE_IAF, VAE_IAF_Config

        if args.model_config is not None:
            model_config = VAE_IAF_Config.from_json_file(args.model_config)

        else:
            model_config = VAE_IAF_Config()

        model_config.input_dim = data_input_dim

        model = VAE_IAF(
            model_config=model_config,
            encoder=Encoder(model_config),
            decoder=Decoder(model_config),
        )

    elif args.model_name == "vae_lin_nf":
        from pythae.models import VAE_LinNF, VAE_LinNF_Config

        if args.model_config is not None:
            model_config = VAE_LinNF_Config.from_json_file(args.model_config)

        else:
            model_config = VAE_LinNF_Config()

        model_config.input_dim = data_input_dim

        model = VAE_LinNF(
            model_config=model_config,
            encoder=Encoder(model_config),
            decoder=Decoder(model_config),
        )

        print(model)

    logger.info(f"Successfully build {args.model_name.upper()} model !\n")

    encoder_num_param = sum(
        p.numel() for p in model.encoder.parameters() if p.requires_grad
    )
    decoder_num_param = sum(
        p.numel() for p in model.decoder.parameters() if p.requires_grad
    )
    total_num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "----------------------------------------------------------------------"
    )
    logger.info("Model \t Encoder params \t Decoder params \t Total params")
    logger.info(
        f"{args.model_name.upper()} \t {encoder_num_param} \t \t {decoder_num_param}"
        f" \t \t {total_num_param}"
    )
    logger.info(
        "----------------------------------------------------------------------\n"
    )

    logger.info(f"Model config of {args.model_name.upper()}: {model_config}\n")

    training_config = AdversarialTrainerConfig.from_json_file(args.training_config)

    # reset the `output_dir` to allow job-arrays
    slurm_array_task_id = 0#int(os.environ["SLURM_ARRAY_TASK_ID"])

    training_config.output_dir = os.path.join(
        f"reproducibility/{args.dataset}",f"{model.model_name}".lower(), f"{slurm_array_task_id}"
    )

    logger.info(f"Training config: {training_config}\n")

    callbacks = []

    if args.use_wandb:
        from pythae.trainers.training_callbacks import WandbCallback

        wandb_cb = WandbCallback()
        wandb_cb.setup(
            training_config,
            model_config=model_config,
            project_name=args.wandb_project,
            entity_name=args.wandb_entity,
        )

        callbacks.append(wandb_cb)

    ### Process data
    data_processor = DataProcessor()
    logger.info("Preprocessing train data...")
    #train_data = data_processor.process_data(train_data).reshape(-1, 1, 64, 64)
    train_dataset = data_processor.to_dataset(train_data)

    logger.info("Preprocessing eval data...\n")
    #eval_data = data_processor.process_data(eval_data)
    eval_dataset = data_processor.to_dataset(eval_data)

    ### Optimizers
    import itertools
    optimizer_enc = torch.optim.Adam(
            itertools.chain(model.encoder.parameters(), model.decoder.parameters()),
            lr=training_config.learning_rate,
            weight_decay=training_config.autoencoder_optim_decay,
        )
    optimizer_dis = torch.optim.Adam(
            model.discriminator.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.discriminator_optim_decay,
        )
    ### Scheduler
    #scheduler_enc = torch.optim.lr_scheduler.MultiStepLR(
    #    optimizer, milestones=[10, 30, 90, 270, 810, 2430, 7290, 21870], gamma=10**(-1/7), verbose=True
    #)

    seed = 123
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    logger.info("Using Base Trainer\n")
    trainer = AdversarialTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=None,#eval_dataset,
        training_config=training_config,
        autoencoder_optimizer=optimizer_enc,
        discriminator_optimizer=optimizer_dis,
        autoencoder_scheduler=None,
        discriminator_scheduler=None,
        callbacks=callbacks,
    )

    print(trainer.scheduler)

    trainer.train()
    
    trained_model = AutoModel.load_from_folder(os.path.join(training_config.output_dir, f'{trainer.model.model_name}_training_{pipeline.trainer._training_signature}', 'final_model')).to(device)

    test_data = torch.tensor(test_data).to(device).type(torch.float)

    with torch.no_grad():
        nll = []
        for i in range(5):
            nll_i = trained_model.get_nll(test_data[:10], n_samples=200)
            logger.info(f"Round {i+1} nll: {nll_i}")
            nll.append(nll_i)
    
    logger.info(
        f'\nmean_nll: {np.mean(nll)}'
    )
    logger.info(
        f'\std_nll: {np.std(nll)}'
    )

if __name__ == "__main__":

    main(args)
