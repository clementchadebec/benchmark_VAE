import argparse
import logging
import os
from typing import List

import numpy as np
import torch

from pythae.data.preprocessors import DataProcessor
from pythae.models import AutoModel
from pythae.models import PIWAE, PIWAEConfig
from pythae.trainers import CoupledOptimizerTrainerConfig, CoupledOptimizerTrainer
from pythae.data.datasets import DatasetOutput
from torch.utils.data import Dataset


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

args = ap.parse_args()

def unif_init(m, n_in, n_out):
    scale = np.sqrt(6./(n_in+n_out))
    m.weight.data.uniform_(
            -scale, scale
        )
    m.bias.data = torch.zeros((1, n_out))

### Define paper encoder network
class Encoder(BaseEncoder):
    def __init__(self, args: dict):
        BaseEncoder.__init__(self)
        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim


        self.fc1 = nn.Linear(np.prod(args.input_dim), 200)
        self.fc2 = nn.Linear(200, 200)

        self.embedding = nn.Linear(200, self.latent_dim)
        self.log_var = nn.Linear(200, self.latent_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                unif_init(m, m.in_features, m.out_features)

    def forward(self, x, output_layer_levels: List[int] = None):
        output = ModelOutput()

        out = torch.tanh(self.fc1(x.reshape(x.shape[0], -1)))
        out = torch.tanh(self.fc2(out))
        output["embedding"] = self.embedding(out)
        output["log_covariance"] = self.log_var(out)

        return output

class DynBinarizedMNIST(Dataset):
    def __init__(self, data):
        self.data = data.type(torch.float)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        x = (x > torch.distributions.Uniform(0, 1).sample(x.shape)).float()
        return DatasetOutput(data=x)

### Define paper decoder network
class Decoder(BaseDecoder):
    def __init__(self, args: dict):
        BaseDecoder.__init__(self)

        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim

        self.fc1 = nn.Linear(self.latent_dim, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, np.prod(args.input_dim))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                unif_init(m, m.in_features, m.out_features)

    def forward(self, z: torch.Tensor, output_layer_levels: List[int] = None):

        output = ModelOutput()

        out = torch.tanh(self.fc1(z))
        out = torch.tanh(self.fc2(out))
        out = torch.sigmoid(self.fc3(out))

        output["reconstruction"] = out.reshape((z.shape[0],) + self.input_dim)

        return output



def main(args):

    
    
    ### Load data
    train_data = torch.tensor(
        np.load(os.path.join(PATH, f"data/mnist", "train_data.npz"))[
            "data"
        ]
        / 255.0
    )
    eval_data = torch.tensor(
        np.load(os.path.join(PATH, f"data/mnist", "eval_data.npz"))["data"]
        / 255.0
    )

    train_data = torch.cat((train_data, eval_data))

    test_data = (
        np.load(os.path.join(PATH, f"data/mnist", "test_data.npz"))["data"]
        / 255.0
    )
    data_input_dim = tuple(train_data.shape[1:])

    if args.model_config is not None:
        model_config = PIWAEConfig.from_json_file(args.model_config)

    else:
        model_config = PIWAEConfig()

    model_config.input_dim = data_input_dim

    model = PIWAE(
        model_config=model_config,
        encoder=Encoder(model_config),
        decoder=Decoder(model_config),
    )

    ### Set training config
    training_config = CoupledOptimizerTrainerConfig.from_json_file(args.training_config)

    logger.info("Preprocessing train data...")
    train_dataset = DynBinarizedMNIST(train_data)

    ### Optimizers
    enc_optimizer = torch.optim.Adam(model.encoder.parameters(), lr=training_config.learning_rate, eps=1e-4)
    dec_optimizer = torch.optim.Adam(model.decoder.parameters(), lr=training_config.learning_rate, eps=1e-4)

    ### Schedulers
    enc_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        enc_optimizer, milestones=[2, 5, 14, 28, 41, 122, 365, 1094], gamma=10**(-1/7), verbose=True
    )
    dec_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        dec_optimizer, milestones=[2, 5, 14, 28, 41, 122, 365, 1094], gamma=10**(-1/7), verbose=True
    )

    seed = 123
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    trainer = CoupledOptimizerTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=None,#eval_dataset,
        training_config=training_config,
        encoder_optimizer=enc_optimizer,
        decoder_optimizer=dec_optimizer,
        encoder_scheduler=enc_scheduler,
        decoder_scheduler=dec_scheduler,
        callbacks=None,
    )

    trainer.train()

    ### Reload model
    trained_model = AutoModel.load_from_folder(os.path.join(training_config.output_dir, f'{trainer.model.model_name}_training_{trainer._training_signature}', 'final_model')).to(device).eval()

    test_data = torch.tensor(test_data).to(device).type(torch.float)
    test_data = (test_data > torch.distributions.Uniform(0, 1).sample(test_data.shape).to(test_data.device)).float()

    ### Compute NLL
    with torch.no_grad():
        nll = []
        for i in range(5):
            nll_i = trained_model.get_nll(test_data, n_samples=5000, batch_size=5000)
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
