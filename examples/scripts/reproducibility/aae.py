import logging
import os
from typing import List

import numpy as np
import torch
import torch.nn as nn

from pythae.data.preprocessors import DataProcessor
from pythae.models import Adversarial_AE, Adversarial_AE_Config
from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn import BaseDecoder, BaseDiscriminator, BaseEncoder
from pythae.trainers import AdversarialTrainer, AdversarialTrainerConfig

logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)

PATH = os.path.dirname(os.path.abspath(__file__))

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
        self.log_var = nn.Linear(1024 * 4 * 4, args.latent_dim)

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
                output["log_covariance"] = self.log_var(out.reshape(x.shape[0], -1))
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
                nn.ConvTranspose2d(128, self.n_channels, 5, 1, padding=1), nn.Sigmoid()
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


### Define paper discriminator network
class Discriminator(BaseDiscriminator):
    def __init__(self, args: dict):
        BaseDiscriminator.__init__(self)

        self.discriminator_input_dim = args.latent_dim

        layers = nn.ModuleList()

        layers.append(
            nn.Sequential(
                nn.Linear(np.prod(self.discriminator_input_dim), 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
            )
        )

        layers.append(nn.Sequential(nn.Linear(512, 1), nn.Sigmoid()))

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

        out = z.reshape(z.shape[0], -1)

        for i in range(max_depth):
            out = self.layers[i](out)

            if output_layer_levels is not None:
                if i + 1 in output_layer_levels:
                    output[f"embedding_layer_{i+1}"] = out

            if i + 1 == self.depth:
                output["embedding"] = out

        return output


def main():

    train_data = (
        np.load(os.path.join(PATH, f"data/celeba", "train_data.npz"))["data"] / 255.0
    )

    data_input_dim = tuple(train_data.shape[1:])

    model_config = Adversarial_AE_Config(
        input_dim=data_input_dim,
        latent_dim=64,
        reconstruction_loss="mse",
        adversarial_loss_scale=0.5,
        reconstruction_loss_scale=0.05,
        deterministic_posterior=True,
    )

    model = Adversarial_AE(
        model_config=model_config,
        encoder=Encoder(model_config),
        decoder=Decoder(model_config),
        discriminator=Discriminator(model_config),
    )

    ### Set training config
    training_config = AdversarialTrainerConfig(
        output_dir="my_models_on_celeba",
        per_device_train_batch_size=100,
        per_device_eval_batch_size=100,
        num_epochs=100,
        autoencoder_learning_rate=3e-4,
        discriminator_learning_rate=1e-3,
        steps_saving=3,
        steps_predict=1000,
        no_cuda=False,
        autoencoder_scheduler_cls="LambdaLR",
        autoencoder_scheduler_params={
            "lr_lambda": lambda epoch: 1 * (epoch < 30)
            + 0.5 * (30 <= epoch < 50)
            + 0.2 * (50 <= epoch),
            "verbose": True,
        },
        discriminator_scheduler_cls="LambdaLR",
        discriminator_scheduler_params={
            "lr_lambda": lambda epoch: 1 * (epoch < 30)
            + 0.5 * (30 <= epoch < 50)
            + 0.2 * (50 <= epoch),
            "verbose": True,
        },
    )

    ### Process data
    data_processor = DataProcessor()
    logger.info("Preprocessing train data...")
    train_data = data_processor.process_data(train_data)
    train_dataset = data_processor.to_dataset(train_data)

    seed = 123
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    logger.info("Using Base Trainer\n")
    trainer = AdversarialTrainer(
        model=model,
        train_dataset=train_dataset,
        training_config=training_config,
        callbacks=None,
    )

    trainer.train()


if __name__ == "__main__":

    main()
