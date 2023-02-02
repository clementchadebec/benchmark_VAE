import logging
import os
from typing import List

import numpy as np
import torch
import torch.nn as nn

from pythae.data.preprocessors import DataProcessor
from pythae.models import BetaTCVAE, BetaTCVAEConfig
from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn import BaseDecoder, BaseEncoder
from pythae.trainers import BaseTrainer, BaseTrainerConfig

logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)

PATH = os.path.dirname(os.path.abspath(__file__))

### Define paper encoder network
class Encoder(BaseEncoder):
    def __init__(self, args: dict):
        super(Encoder, self).__init__()
        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim

        layers = nn.ModuleList()

        layers.append(
            nn.Sequential(
                nn.Linear(np.prod(args.input_dim), 1200),
                nn.ReLU(inplace=True),
                nn.Linear(1200, 1200),
                nn.ReLU(inplace=True),
            )
        )

        self.embedding = nn.Linear(1200, self.latent_dim)
        self.log_var = nn.Linear(1200, self.latent_dim)

        self.layers = layers
        self.depth = len(layers)

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
                output["log_covariance"] = self.log_var(out)

        return output


### Define paper decoder network
class Decoder(BaseDecoder):
    def __init__(self, args: dict):
        BaseDecoder.__init__(self)

        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim

        layers = nn.ModuleList()

        layers = nn.Sequential(
            nn.Linear(self.latent_dim, 1200),
            nn.Tanh(),
            nn.Linear(1200, 1200),
            nn.Tanh(),
            nn.Linear(1200, 1200),
            nn.Tanh(),
            nn.Linear(1200, np.prod(args.input_dim)),
            nn.Sigmoid(),
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


def main():

    data = np.load(
        os.path.join(
            PATH, f"data/dsprites", "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
        ),
        encoding="latin1",
    )
    train_data = torch.from_numpy(data["imgs"]).float()

    data_input_dim = tuple(train_data.shape[1:])

    model_config = BetaTCVAEConfig(
        input_dim=data_input_dim,
        latent_dim=10,
        reconstruction_loss="bce",
        beta=6,
        gamma=1,
        alpha=1,
        use_mss=False,
    )

    model = BetaTCVAE(
        model_config=model_config,
        encoder=Encoder(model_config),
        decoder=Decoder(model_config),
    )

    ### Set the training config
    training_config = BaseTrainerConfig(
        output_dir="reproducibility/dsprites",
        per_device_train_batch_size=1000,
        per_device_eval_batch_size=1000,
        num_epochs=50,
        learning_rate=1e-3,
        steps_saving=None,
        steps_predict=None,
        no_cuda=False,
        optimizer_cls="Adam",
        optimizer_params={"eps": 1e-4},
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
    trainer = BaseTrainer(
        model=model,
        train_dataset=train_dataset,
        training_config=training_config,
        callbacks=None,
    )

    trainer.train()


if __name__ == "__main__":

    main()
