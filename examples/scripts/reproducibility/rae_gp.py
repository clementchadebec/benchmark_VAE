import argparse
import logging
import os
from typing import List

import numpy as np
import torch

from pythae.data.preprocessors import DataProcessor
from pythae.trainers import BaseTrainer, BaseTrainerConfig
from pythae.models import RAE_GP, RAE_GP_Config
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


### Define paper encoder network
class Encoder(BaseEncoder):
    def __init__(self, args):
        BaseEncoder.__init__(self)

        self.input_dim = (3, 64, 64)
        self.latent_dim = args.latent_dim
        self.n_channels = 3

        layers = nn.ModuleList()

        layers.append(
            nn.Sequential(
                nn.Conv2d(self.n_channels, 128, 4, 2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.Conv2d(128, 256, 4, 2, padding=1), nn.BatchNorm2d(256), nn.ReLU()
            )
        )

        layers.append(
            nn.Sequential(
                nn.Conv2d(256, 512, 4, 2, padding=1), nn.BatchNorm2d(512), nn.ReLU()
            )
        )

        layers.append(
            nn.Sequential(
                nn.Conv2d(512, 1024, 4, 2, padding=1), nn.BatchNorm2d(1024), nn.ReLU()
            )
        )

        self.layers = layers
        self.depth = len(layers)

        self.embedding = nn.Linear(1024 * 4 * 4, args.latent_dim)

    def forward(self, x: torch.Tensor, output_layer_levels: List[int] = None):
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

        out = x

        for i in range(max_depth):
            out = self.layers[i](out)

            if output_layer_levels is not None:
                if i + 1 in output_layer_levels:
                    output[f"embedding_layer_{i+1}"] = out
            if i + 1 == self.depth:
                output["embedding"] = self.embedding(out.reshape(x.shape[0], -1))

        return output


### Define paper decoder network
class Decoder(BaseDecoder):
    def __init__(self, args: dict):
        BaseDecoder.__init__(self)
        self.input_dim = (3, 64, 64)
        self.latent_dim = args.latent_dim
        self.n_channels = 3

        layers = nn.ModuleList()

        layers.append(nn.Sequential(nn.Linear(args.latent_dim, 1024 * 8 * 8)))

        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(1024, 512, 5, 2, padding=2),
                nn.BatchNorm2d(512),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(512, 256, 5, 2, padding=1, output_padding=0),
                nn.BatchNorm2d(256),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(256, 128, 5, 2, padding=2, output_padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(128, self.n_channels, 5, 1, padding=1),
                nn.Sigmoid()
            )
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

            if i == 0:
                out = out.reshape(z.shape[0], 1024, 8, 8)

            if output_layer_levels is not None:
                if i + 1 in output_layer_levels:
                    output[f"reconstruction_layer_{i+1}"] = out

            if i + 1 == self.depth:
                output["reconstruction"] = out

        return output



def main(args):

    train_data = (
        np.load(os.path.join(PATH, f"data/celeba", "train_data.npz"))[
            "data"
        ]
        / 255.0
    )
    eval_data = (
        np.load(os.path.join(PATH, f"data/celeba", "eval_data.npz"))["data"]
        / 255.0
    )

    data_input_dim = tuple(train_data.shape[1:])

    if args.model_config is not None:
        model_config = RAE_GP_Config.from_json_file(args.model_config)

    else:
        model_config = RAE_GP_Config()

    model_config.input_dim = data_input_dim

    model = RAE_GP(
        model_config=model_config,
        encoder=Encoder(model_config),
        decoder=Decoder(model_config),
    )

    ### Set training config
    training_config = BaseTrainerConfig.from_json_file(args.training_config)

    ### Process data
    data_processor = DataProcessor()
    logger.info("Preprocessing train data...")
    train_data = data_processor.process_data(train_data)
    train_dataset = data_processor.to_dataset(train_data)

    logger.info("Preprocessing eval data...\n")
    eval_data = data_processor.process_data(eval_data)
    eval_dataset = data_processor.to_dataset(eval_data)

    ### Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config.learning_rate)

    ### Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=5, verbose=True
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

if __name__ == "__main__":

    main(args)
