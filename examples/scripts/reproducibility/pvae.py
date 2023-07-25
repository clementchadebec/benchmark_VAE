import logging
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pythae.data.preprocessors import DataProcessor
from pythae.models import AutoModel, PoincareVAE, PoincareVAEConfig
from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn import BaseDecoder, BaseEncoder
from pythae.models.pvae.pvae_utils import PoincareBall
from pythae.trainers import BaseTrainer, BaseTrainerConfig

logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)

PATH = os.path.dirname(os.path.abspath(__file__))

device = "cuda" if torch.cuda.is_available() else "cpu"


class RiemannianLayer(nn.Module):
    def __init__(self, in_features, out_features, manifold, over_param, weight_norm):
        super(RiemannianLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.manifold = manifold
        self._weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.over_param = over_param
        self.weight_norm = weight_norm
        self._bias = nn.Parameter(torch.Tensor(out_features, 1))
        self.reset_parameters()

    @property
    def weight(self):
        return self.manifold.transp0(
            self.bias, self._weight
        )  # weight \in T_0 => weight \in T_bias

    @property
    def bias(self):
        if self.over_param:
            return self._bias
        else:
            return self.manifold.expmap0(
                self._weight * self._bias
            )  # reparameterisation of a point on the manifold

    def reset_parameters(self):
        nn.init.kaiming_normal_(self._weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self._weight)
        bound = 4 / math.sqrt(fan_in)
        nn.init.uniform_(self._bias, -bound, bound)
        if self.over_param:
            with torch.no_grad():
                self._bias.set_(self.manifold.expmap0(self._bias))


class GeodesicLayer(RiemannianLayer):
    def __init__(
        self, in_features, out_features, manifold, over_param=False, weight_norm=False
    ):
        super(GeodesicLayer, self).__init__(
            in_features, out_features, manifold, over_param, weight_norm
        )

    def forward(self, input):
        input = input.unsqueeze(0)
        input = input.unsqueeze(-2).expand(
            *input.shape[: -(len(input.shape) - 2)], self.out_features, self.in_features
        )
        res = self.manifold.normdist2plane(
            input, self.bias, self.weight, signed=True, norm=self.weight_norm
        )
        return res


### Define paper encoder network
class Encoder(BaseEncoder):
    """Usual encoder followed by an exponential map"""

    def __init__(self, model_config, prior_iso):
        super(Encoder, self).__init__()
        self.manifold = PoincareBall(
            dim=model_config.latent_dim, c=model_config.curvature
        )
        self.enc = nn.Sequential(
            nn.Linear(np.prod(model_config.input_dim), 600),
            nn.ReLU(),
        )
        self.fc21 = nn.Linear(600, model_config.latent_dim)
        self.fc22 = nn.Linear(600, model_config.latent_dim if not prior_iso else 1)

    def forward(self, x):
        e = self.enc(x.reshape(x.shape[0], -1))
        mu = self.fc21(e)
        mu = self.manifold.expmap0(mu)
        return ModelOutput(
            embedding=mu,
            log_covariance=torch.log(
                F.softplus(self.fc22(e)) + 1e-5
            ),  # expects log_covariance
            log_concentration=torch.log(
                F.softplus(self.fc22(e)) + 1e-5
            ),  # for Riemannian Normal
        )


### Define paper decoder network
class Decoder(BaseDecoder):
    """First layer is a Hypergyroplane followed by usual decoder"""

    def __init__(self, model_config):
        super(Decoder, self).__init__()
        self.manifold = PoincareBall(
            dim=model_config.latent_dim, c=model_config.curvature
        )
        self.input_dim = model_config.input_dim
        self.dec = nn.Sequential(
            GeodesicLayer(model_config.latent_dim, 600, self.manifold),
            nn.ReLU(),
            nn.Linear(600, np.prod(model_config.input_dim)),
            nn.Sigmoid(),
        )

    def forward(self, z):
        out = self.dec(z).reshape((z.shape[0],) + self.input_dim)  # reshape data
        return ModelOutput(reconstruction=out)


def main():

    ### Load data
    train_data = torch.tensor(
        np.load(os.path.join(PATH, f"data/mnist", "train_data.npz"))["data"] / 255.0
    ).clamp(1e-5, 1 - 1e-5)
    eval_data = torch.tensor(
        np.load(os.path.join(PATH, f"data/mnist", "eval_data.npz"))["data"] / 255.0
    ).clamp(1e-5, 1 - 1e-5)

    train_data = torch.cat((train_data, eval_data))

    test_data = torch.tensor(
        np.load(os.path.join(PATH, f"data/mnist", "test_data.npz"))["data"] / 255.0
    ).clamp(1e-5, 1 - 1e-5)

    data_input_dim = tuple(train_data.shape[1:])

    model_config = PoincareVAEConfig(
        input_dim=data_input_dim,
        latent_dim=10,
        reconstruction_loss="bce",
        prior_distribution="wrapped_normal",
        posterior_distribution="wrapped_normal",
        curvature=0.7,
    )

    model = PoincareVAE(
        model_config=model_config,
        encoder=Encoder(model_config, prior_iso=True),
        decoder=Decoder(model_config),
    )

    ### Set training config
    training_config = BaseTrainerConfig(
        output_dir="reproducibility/mnist",
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        num_epochs=80,
        learning_rate=5e-4,
        steps_saving=100,
        steps_predict=None,
        no_cuda=False,
    )

    ### Process data
    data_processor = DataProcessor()
    logger.info("Preprocessing train data...")
    train_data = data_processor.process_data(torch.bernoulli(train_data))
    train_dataset = data_processor.to_dataset(train_data)

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
