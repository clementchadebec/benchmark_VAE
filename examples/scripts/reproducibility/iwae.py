import argparse
import logging
import os
from typing import List

import numpy as np
import torch

from pythae.data.preprocessors import DataProcessor
from pythae.models import AutoModel
from pythae.models import IWAE, IWAEConfig
from pythae.trainers import BaseTrainer, BaseTrainerConfig

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

        out = torch.tanh(self.fc1(x))
        out = torch.tanh(self.fc2(out))
        output["embedding"] = self.embedding(out)
        output["log_covariance"] = self.log_var(out)

        return output

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

    
    
    train_data = torch.tensor(np.loadtxt(os.path.join(PATH, f"data/binary_mnist", "binarized_mnist_train.amat"))).type(torch.float)
    eval_data = torch.tensor(np.loadtxt(os.path.join(PATH, f"data/binary_mnist", "binarized_mnist_valid.amat"))).type(torch.float)
    test_data = torch.tensor(np.loadtxt(os.path.join(PATH, f"data/binary_mnist", "binarized_mnist_test.amat"))).type(torch.float)

    data_input_dim = tuple(train_data.shape[1:])

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

    ### Set training config
    training_config = BaseTrainerConfig.from_json_file(args.training_config)

    ### Process data
    data_processor = DataProcessor()
    logger.info("Preprocessing train data...")
    #train_data = data_processor.process_data(train_data)
    train_dataset = data_processor.to_dataset(train_data)

    logger.info("Preprocessing eval data...\n")
    #ieval_data = data_processor.process_data(eval_data)
    eval_dataset = data_processor.to_dataset(eval_data)

    ### Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config.learning_rate, eps=1e-4)

    ### Scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[2, 5, 14, 28, 41, 122, 365, 1094], gamma=10**(-1/7), verbose=True
    )

    seed = 123
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    logger.info("Using Base Trainer\n")
    trainer = BaseTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        training_config=training_config,
        optimizer=optimizer,
        scheduler=scheduler,
        callbacks=None,
    )

    trainer.train()

    ### Reload model
    trained_model = AutoModel.load_from_folder(os.path.join(training_config.output_dir, f'{trainer.model.model_name}_training_{trainer._training_signature}', 'final_model')).to(device).eval()

    test_data = torch.tensor(test_data).to(device).type(torch.float)

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
