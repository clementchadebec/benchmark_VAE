import argparse
import logging
import os

import torch

from pyraug.models import RHVAE
from pyraug.models.rhvae.rhvae_sampler import RHVAESampler, RHVAESamplerConfig

PATH = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()

# Generation setting
ap.add_argument(
    "--num_samples", type=int, help="the number of samples to generate", required=True
)
ap.add_argument(
    "--path_to_model_folder",
    type=str,
    help="path to the model from which to generate",
    required=True,
)
ap.add_argument(
    "--path_to_sampler_config",
    type=str,
    help="path to the sampler config",
    default=os.path.join(PATH, "configs/rhvae_sampler_config.json"),
)

args = ap.parse_args()


def main(args):

    sampler_config = RHVAESamplerConfig.from_json_file(args.path_to_sampler_config)
    model = RHVAE.load_from_folder(args.path_to_model_folder)
    sampler = RHVAESampler(model=model, sampler_config=sampler_config)

    sampler.sample(args.num_samples)


if __name__ == "__main__":

    main(args)
