import logging
import os
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from pythae.data.datasets import DatasetOutput
from pythae.models import SVAE, AutoModel, SVAEConfig
from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn import BaseDecoder, BaseEncoder
from pythae.trainers import BaseTrainer, BaseTrainerConfig

logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)

PATH = os.path.dirname(os.path.abspath(__file__))

device = "cuda" if torch.cuda.is_available() else "cpu"

### Define paper encoder network
class Encoder(BaseEncoder):
    def __init__(self, args: dict):
        BaseEncoder.__init__(self)
        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim

        layers = nn.ModuleList()

        layers.append(
            nn.Sequential(
                nn.Linear(np.prod(args.input_dim), 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
            )
        )

        self.layers = layers
        self.depth = len(layers)

        self.embedding = nn.Linear(128, self.latent_dim)
        self.log_concentration = nn.Linear(128, 1)

    def forward(self, x, output_layer_levels: List[int] = None):
        output = ModelOutput()

        max_depth = self.depth

        if output_layer_levels is not None:

            assert all(
                self.depth >= levels > 0 or levels == -1
                for levels in output_layer_levels
            ), f"Cannot output layer deeper than depth ({self.depth}). Got ({output_layer_levels})."

            if -1 in output_layer_levels:
                max_depth = self.depth
            else:
                max_depth = max(output_layer_levels)

        out = x.reshape(-1, np.prod(self.input_dim))

        for i in range(max_depth):
            out = self.layers[i](out)

            if output_layer_levels is not None:
                if i + 1 in output_layer_levels:
                    output[f"embedding_layer_{i+1}"] = out
            if i + 1 == self.depth:
                output["embedding"] = self.embedding(out)
                output["log_concentration"] = self.log_concentration(out)

        return output


### Define paper decoder network
class Decoder(BaseDecoder):
    def __init__(self, args: dict):
        BaseDecoder.__init__(self)

        self.input_dim = args.input_dim

        layers = nn.ModuleList()

        layers.append(
            nn.Sequential(
                nn.Linear(args.latent_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(nn.Linear(256, np.prod(args.input_dim)), nn.Sigmoid())
        )

        self.layers = layers
        self.depth = len(layers)

    def forward(self, z: torch.Tensor, output_layer_levels: List[int] = None):

        output = ModelOutput()

        max_depth = self.depth

        if output_layer_levels is not None:

            assert all(
                self.depth >= levels > 0 or levels == -1
                for levels in output_layer_levels
            ), (
                f"Cannot output layer deeper than depth ({self.depth}). "
                f"Got ({output_layer_levels})."
            )

            if -1 in output_layer_levels:
                max_depth = self.depth
            else:
                max_depth = max(output_layer_levels)

        out = z

        for i in range(max_depth):
            out = self.layers[i](out)

            if output_layer_levels is not None:
                if i + 1 in output_layer_levels:
                    output[f"reconstruction_layer_{i+1}"] = out
            if i + 1 == self.depth:
                output["reconstruction"] = out.reshape((z.shape[0],) + self.input_dim)

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


def main():

    ### Load data
    train_data = torch.tensor(
        np.load(os.path.join(PATH, f"data/mnist", "train_data.npz"))["data"] / 255.0
    )
    eval_data = torch.tensor(
        np.load(os.path.join(PATH, f"data/mnist", "eval_data.npz"))["data"] / 255.0
    )

    train_data = torch.cat((train_data, eval_data))

    test_data = (
        np.load(os.path.join(PATH, f"data/mnist", "test_data.npz"))["data"] / 255.0
    )

    data_input_dim = tuple(train_data.shape[1:])

    model_config = SVAEConfig(
        input_dim=data_input_dim, latent_dim=11, reconstruction_loss="bce"
    )

    model = SVAE(
        model_config=model_config,
        encoder=Encoder(model_config),
        decoder=Decoder(model_config),
    )

    ### Set training config
    training_config = BaseTrainerConfig(
        output_dir="reproducibility/binary_mnist",
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        num_epochs=500,
        learning_rate=1e-3,
        steps_saving=50,
        steps_predict=None,
        no_cuda=False,
    )

    logger.info("Preprocessing train data...")
    train_dataset = DynBinarizedMNIST(train_data)

    seed = 123
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    logger.info("Using Base Trainer\n")
    trainer = BaseTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=None,  # eval_dataset,
        training_config=training_config,
        callbacks=None,
    )

    ### Launch training
    trainer.train()

    trained_model = (
        AutoModel.load_from_folder(
            os.path.join(
                training_config.output_dir,
                f"{trainer.model.model_name}_training_{trainer._training_signature}",
                "final_model",
            )
        )
        .to(device)
        .eval()
    )

    test_data = torch.tensor(test_data).to(device).type(torch.float)
    test_data = (
        test_data
        > torch.distributions.Uniform(0, 1).sample(test_data.shape).to(test_data.device)
    ).float()

    ### Compute NLL
    with torch.no_grad():
        nll = []
        for i in range(5):
            nll_i = trained_model.get_nll(test_data, n_samples=500, batch_size=500)
            logger.info(f"Round {i+1} nll: {nll_i}")
            nll.append(nll_i)

    logger.info(f"\nmean_nll: {np.mean(nll)}")
    logger.info(f"\std_nll: {np.std(nll)}")


if __name__ == "__main__":

    main()
