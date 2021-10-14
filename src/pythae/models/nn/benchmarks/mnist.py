import torch
import torch.nn as nn

from ..base_architectures import BaseEncoder, BaseDecoder
from ....models.base.base_utils import ModelOuput
from ....models import BaseAEConfig


class Encoder_AE_MNIST(BaseEncoder):

    def __init__(self, args: BaseAEConfig):
        BaseEncoder.__init__(self)

        self.input_dim = (1, 28, 28)
        self.latent_dim = args.latent_dim
        self.n_channels = 1
        
        self.conv_layers = nn.Sequential(
                        nn.Conv2d(self.n_channels, 128, 4, 2, padding=1),
                        nn.BatchNorm2d(128),
                        nn.ReLU(),
                        nn.Conv2d(128, 256, 4, 2, padding=1),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                        nn.Conv2d(256, 512, 4, 2, padding=1),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512, 1024, 4, 2, padding=1),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    )

        self.embedding = nn.Linear(1024, args.latent_dim)

    def forward(self, x: torch.Tensor):
        h1 = self.conv_layers(x).reshape(x.shape[0], -1)
        output = ModelOuput(
            embedding=self.embedding(h1)
        )
        return output

class Encoder_VAE_MNIST(BaseEncoder):

    def __init__(self, args: BaseAEConfig):
        BaseEncoder.__init__(self)

        self.input_dim = (1, 28, 28)
        self.latent_dim = args.latent_dim
        self.n_channels = 1
        
        self.conv_layers = nn.Sequential(
                        nn.Conv2d(self.n_channels, 128, 4, 2, padding=1),
                        nn.BatchNorm2d(128),
                        nn.ReLU(),
                        nn.Conv2d(128, 256, 4, 2, padding=1),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                        nn.Conv2d(256, 512, 4, 2, padding=1),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512, 1024, 4, 2, padding=1),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    )

        self.embedding = nn.Linear(1024, args.latent_dim)
        self.log_var =  nn.Linear(1024, args.latent_dim)

    def forward(self, x: torch.Tensor):
        h1 = self.conv_layers(x).reshape(x.shape[0], -1)
        output = ModelOuput(
            embedding=self.embedding(h1),
            log_covariance=self.log_var(h1)
        )
        return output


class Decoder_AE_MNIST(BaseDecoder):
    def __init__(self, args: dict):
        BaseDecoder.__init__(self)
        self.input_dim = (1, 28, 28)
        self.latent_dim = args.latent_dim
        self.n_channels = 1

        self.fc = nn.Linear(args.latent_dim, 1024*4*4)
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 3, 2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 3, 2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, self.n_channels, 3, 2, padding=1, output_padding=1),
            nn.Sigmoid()
            )
    
    def forward(self, z: torch.Tensor):
        h1 = self.fc(z).reshape(z.shape[0], 1024, 4, 4)
        output = ModelOuput(
            reconstruction=self.deconv_layers(h1)
        )
    
        return output
