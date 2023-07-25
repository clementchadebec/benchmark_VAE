import logging
import os
from typing import List

import numpy as np
import torch
import torch.nn as nn

from pythae.data.preprocessors import DataProcessor
from pythae.models import HVAE, AutoModel, HVAEConfig
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
            nn.Sequential(nn.Linear(np.prod(args.input_dim), 300), nn.Softplus())
        )

        self.layers = layers
        self.depth = len(layers)

        self.embedding = nn.Linear(300, self.latent_dim)
        self.log_var = nn.Linear(300, self.latent_dim)

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

        layers = nn.ModuleList()

        layers.append(nn.Sequential(nn.Linear(args.latent_dim, 300), nn.Softplus()))

        layers.append(
            nn.Sequential(
                nn.Linear(300, np.prod(args.input_dim)),
                # nn.Sigmoid()
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

            if output_layer_levels is not None:
                if i + 1 in output_layer_levels:
                    output[f"reconstruction_layer_{i+1}"] = out
            if i + 1 == self.depth:
                output["reconstruction"] = out.reshape((z.shape[0],) + self.input_dim)

        return output


def main():

    train_data = np.loadtxt(
        os.path.join(PATH, f"data/binary_mnist", "binarized_mnist_train.amat")
    ).reshape(-1, 1, 28, 28)
    eval_data = np.loadtxt(
        os.path.join(PATH, f"data/binary_mnist", "binarized_mnist_valid.amat")
    ).reshape(-1, 1, 28, 28)
    test_data = np.loadtxt(
        os.path.join(PATH, f"data/binary_mnist", "binarized_mnist_test.amat")
    ).reshape(-1, 1, 28, 28)

    data_input_dim = tuple(train_data.shape[1:])

    model_config = HVAEConfig(
        input_dim=data_input_dim,
        latent_dim=64,
        reconstruction_loss="bce",
        n_lf=4,
        eps_lf=0.05,
        beta_zero=1,
        learn_eps_lf=False,
        learn_beta_zero=False,
    )

    model = HVAE(
        model_config=model_config,
        encoder=Encoder(model_config),
        decoder=Decoder(model_config),
    )

    ### Set training config
    training_config = BaseTrainerConfig(
        output_dir="reproducibility/binary_mnist",
        per_device_train_batch_size=100,
        per_device_eval_batch_size=100,
        num_epochs=2000,
        learning_rate=5e-4,
        steps_saving=50,
        steps_predict=None,
        no_cuda=False,
        optimizer_cls="Adamax",
        scheduler_cls="MultiStepLR",
        scheduler_params={
            "milestones": [200, 350, 500, 750, 1000],
            "gamma": 10 ** (-1 / 5),
            "verbose": True,
        },
    )

    ### Process data
    data_processor = DataProcessor()
    logger.info("Preprocessing train data...")
    train_data = data_processor.process_data(train_data)
    train_dataset = data_processor.to_dataset(train_data)

    logger.info("Preprocessing eval data...\n")
    eval_data = data_processor.process_data(eval_data)
    eval_dataset = data_processor.to_dataset(eval_data)

    seed = 123
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    logger.info("Using Base Trainer\n")
    trainer = BaseTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        training_config=training_config,
        callbacks=None,
    )

    trainer.train()

    ### Reload the model
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

    ### Compute NLL
    with torch.no_grad():
        nll = []
        for i in range(5):
            nll_i = trained_model.get_nll(test_data, n_samples=1000)
            logger.info(f"Round {i+1} nll: {nll_i}")
            nll.append(nll_i)

    logger.info(f"\nmean_nll: {np.mean(nll)}")
    logger.info(f"\std_nll: {np.std(nll)}")


if __name__ == "__main__":

    main()
