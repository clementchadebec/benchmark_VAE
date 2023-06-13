import logging
import os
from typing import List

import numpy as np
import torch
import torch.nn as nn

from pythae.data.preprocessors import DataProcessor
from pythae.models import VAMP, AutoModel, VAMPConfig
from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn import BaseDecoder, BaseEncoder
from pythae.trainers import BaseTrainer, BaseTrainerConfig

logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)

PATH = os.path.dirname(os.path.abspath(__file__))

device = "cuda" if torch.cuda.is_available() else "cpu"


def he_init(m):
    s = np.sqrt(2.0 / m.in_features)
    m.weight.data.normal_(0, s)


### Define custom layers from the paper


class NonLinear(nn.Module):
    def __init__(self, input_size, output_size, bias=True, activation=None):
        super(NonLinear, self).__init__()

        self.activation = activation
        self.linear = nn.Linear(int(input_size), int(output_size), bias=bias)

    def forward(self, x):
        h = self.linear(x)
        if self.activation is not None:
            h = self.activation(h)

        return h


class GatedDense(nn.Module):
    def __init__(self, input_size, output_size, activation=None):
        super(GatedDense, self).__init__()

        self.activation = activation
        self.sigmoid = nn.Sigmoid()
        self.h = nn.Linear(input_size, output_size)
        self.g = nn.Linear(input_size, output_size)

    def forward(self, x):
        h = self.h(x)
        if self.activation is not None:
            h = self.activation(self.h(x))

        g = self.sigmoid(self.g(x))

        return h * g


### Define paper encoder network
class Encoder(BaseEncoder):
    def __init__(self, args: dict):
        BaseEncoder.__init__(self)
        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim

        layers = nn.ModuleList()

        layers.append(
            nn.Sequential(
                GatedDense(np.prod(args.input_dim), 300), GatedDense(300, 300)
            )
        )

        self.layers = layers
        self.depth = len(layers)

        self.embedding = nn.Linear(300, self.latent_dim)
        self.log_var = NonLinear(
            300, self.latent_dim, activation=nn.Hardtanh(min_val=-6.0, max_val=2.0)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                he_init(m)

    def forward(self, x, output_layer_levels: List[int] = None):
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

        layers.append(
            nn.Sequential(GatedDense(args.latent_dim, 300), GatedDense(300, 300))
        )

        layers.append(
            nn.Sequential(
                NonLinear(300, np.prod(args.input_dim), activation=nn.Sigmoid())
            )
        )

        self.layers = layers
        self.depth = len(layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                he_init(m)

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
    )
    eval_data = np.loadtxt(
        os.path.join(PATH, f"data/binary_mnist", "binarized_mnist_valid.amat")
    )
    test_data = np.loadtxt(
        os.path.join(PATH, f"data/binary_mnist", "binarized_mnist_test.amat")
    )

    data_input_dim = tuple(train_data.shape[1:])

    ### Build model
    model_config = VAMPConfig(
        input_dim=data_input_dim,
        latent_dim=40,
        reconstruction_loss="bce",
        number_components=500,
        linear_scheduling_steps=100,
    )

    model = VAMP(
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
        learning_rate=1e-4,
        steps_saving=None,
        steps_predict=None,
        no_cuda=False,
        optimizer_cls="RMSprop",
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

    ### Reload model
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
            nll_i = trained_model.get_nll(test_data, n_samples=5000, batch_size=5000)
            logger.info(f"Round {i+1} nll: {nll_i}")
            nll.append(nll_i)

    logger.info(f"\nmean_nll: {np.mean(nll)}")
    logger.info(f"\std_nll: {np.std(nll)}")


if __name__ == "__main__":
    main()
