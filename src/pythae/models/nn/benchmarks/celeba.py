import torch
import torch.nn as nn

from ..base_architectures import BaseEncoder, BaseDecoder
from ....models.base.base_utils import ModelOuput
from ....models import BaseAEConfig


class Encoder_AE_CELEBA(BaseEncoder):

    def __init__(self, args: BaseAEConfig):
        BaseEncoder.__init__(self)

        self.input_dim = (3, 64, 64)
        self.latent_dim = args.latent_dim
        self.n_channels = 3
        
        self.conv_layers = nn.Sequential(
                        nn.Conv2d(self.n_channels, 128, 5, 2, padding=1),
                        nn.BatchNorm2d(128),
                        nn.ReLU(),
                        nn.Conv2d(128, 256, 5, 2, padding=1),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                        nn.Conv2d(256, 512, 5, 2, padding=2),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.Conv2d(512, 1024, 5, 2, padding=2),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                    )


        self.embedding = nn.Linear(1024*4*4, args.latent_dim)

    def forward(self, x: torch.Tensor):
        h1 = self.conv_layers(x).reshape(x.shape[0], -1)
        output = ModelOuput(
            embedding=self.embedding(h1)
        )
        return output

class Encoder_VAE_CELEBA(BaseEncoder):

    def __init__(self, args: BaseAEConfig):
        BaseEncoder.__init__(self)

        self.input_dim = (3, 64, 64)
        self.latent_dim = args.latent_dim
        self.n_channels = 3
        
        self.conv_layers = torch.nn.Sequential(
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

        self.embedding = nn.Linear(1024*4*4, args.latent_dim)
        self.log_var =  nn.Linear(1024*4*4, args.latent_dim)

    def forward(self, x: torch.Tensor):
        h1 = self.conv_layers(x).reshape(x.shape[0], -1)
        output = ModelOuput(
            embedding=self.embedding(h1),
            log_covariance=self.log_var(h1)
        )
        return output


class Decoder_AE_CELEBA(BaseDecoder):
    def __init__(self, args: dict):
        BaseDecoder.__init__(self)
        self.input_dim = (3, 64, 64)
        self.latent_dim = args.latent_dim
        self.n_channels = 3

        self.fc = nn.Linear(args.latent_dim, 1024*8*8)

        self.deconv_layers = nn.Sequential(
                              nn.ConvTranspose2d(1024, 512, 5, 2, padding=2),
                              nn.BatchNorm2d(512),
                              nn.ReLU(),
                              nn.ConvTranspose2d(512, 256, 5, 2, padding=1, output_padding=0),
                              nn.BatchNorm2d(256),
                              nn.ReLU(),
                              nn.ConvTranspose2d(256, 128, 5, 2, padding=2, output_padding=1),
                              nn.BatchNorm2d(128),
                              nn.ReLU(),
                              nn.ConvTranspose2d(128, self.n_channels, 5, 1, padding=1),
                              nn.Sigmoid()
                              )
    
    def forward(self, z: torch.Tensor):
        h1 = self.fc(z).reshape(z.shape[0], 1024, 8, 8)
        output = ModelOuput(
            reconstruction=self.deconv_layers(h1)
        )
    
        return output
