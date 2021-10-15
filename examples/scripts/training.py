import argparse
import importlib
import logging
import os

import numpy as np
import torch

from pythae.data.preprocessors import DataProcessor
from pythae.models import RHVAE
from pythae.models.rhvae import RHVAEConfig
from pythae.trainers import Trainer
from pythae.trainers.training_config import TrainingConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

PATH = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()

# Training setting
ap.add_argument(
    "--dataset",
    type=str,
    default='mnist',
    choices=['mnist', 'cifar10', 'celeba'],
    help="The data set to use to perform training. It must be located in the folder 'data' at the "
    "path 'data/datset_name/' and contrain a 'train_data.npz' file with the data being "
    "under the key 'data'. The data must be in the range [0-255] and shaped with the channel in "
    "first position (im_channel x height x width).",
    required=True,
)
ap.add_argument(
    "--model_name",
    help="The name of the model to train",
    choices=['ae', 'vae', 'beta_vae', 'wae', 'vamp', 'hvae', 'rhvae'],
    required=True
)
ap.add_argument(
    "--path_to_model_config",
    help="path to model config file (expected json file)",
)
ap.add_argument(
    "--path_to_training_config",
    help="path_to_model_config_file (expected json file)",
    default=os.path.join(PATH, "configs/training_config.json"),
)

args = ap.parse_args()

def main(args):

    if args.model_name == 'ae':
        from pythae.model import AE, AEConfig

        model_config = AEConfig.from_json_file(args.path_to_model_config)
        model = AE()

    model_config = RHVAEConfig.from_json_file(args.path_to_model_config)
    training_config = TrainingConfig.from_json_file(args.path_to_training_config)

    train_data = ImageGetterFromFolder.load(args.path_to_train_data)
    data_processor = DataProcessor(data_normalization_type="individual_min_max_scaling")
    train_data = data_processor.process_data(train_data)
    train_dataset = DataProcessor.to_dataset(train_data)

    # set input dimension automatically
    model_config.input_dim = int(np.prod(train_dataset.data.shape[1:]))

    if args.path_to_eval_data is not None:
        eval_data = ImageGetterFromFolder(args.path_to_eval_data)
        eval_data = data_processor.process_data(eval_data)
        eval_dataset = DataProcessor.to_dataset(eval_data)

    else:
        eval_dataset = None

    model = RHVAE(model_config)

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        training_config=training_config,
    )

    trainer.train(log_output_dir=args.path_to_logs)


if __name__ == "__main__":

    main(args)
