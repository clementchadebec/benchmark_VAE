from dataclasses import field
from typing import List

from pydantic.dataclasses import dataclass

from ..vae import VAEConfig


@dataclass
class VAE_LinNF_Config(VAEConfig):
    """VAE with linear Normalizing Flow config class.

    Parameters:
        input_dim (int): The input_data dimension
        latent_dim (int): The latent space dimension. Default: None.
        reconstruction_loss (str): The reconstruction loss to use ['bce', 'mse']. Default: 'mse'
        flows (List[str]):  A list of strings corresponding to the class of each flow to be applied.
            Default: ['Plannar', 'Planar']. Flow choices: ['Planar', 'Radial'].
    """

    flows: List[str] = field(default_factory=lambda: ["Planar", "Planar"])

    def __post_init__(self):
        super().__post_init__()
        for i, f in enumerate(self.flows):
            assert f in ["Planar", "Radial"], (
                f"Flow name number {i+1}: '{f}' doesn't correspond "
                "to ones of the classes. Available linear flows ['Planar', 'Radial']"
            )
