import argparse
import logging
import os
import time

import hostlist
import numpy as np
import torch
from torch.utils.data import Dataset

from pythae.data.datasets import DatasetOutput
from pythae.models import VQVAE, VQVAEConfig
from pythae.models.nn.benchmarks.mnist import (
    Decoder_ResNet_VQVAE_MNIST,
    Encoder_ResNet_VQVAE_MNIST,
)
from pythae.trainers import BaseTrainer, BaseTrainerConfig

logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)

PATH = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()

# Training setting
ap.add_argument(
    "--use_wandb",
    help="whether to log the metrics in wandb",
    action="store_true",
)
ap.add_argument(
    "--wandb_project",
    help="wandb project name",
    default="mnist-distributed",
)
ap.add_argument(
    "--wandb_entity",
    help="wandb entity name",
    default="clementchadebec",
)

args = ap.parse_args()


class MNIST(Dataset):
    def __init__(self, data):
        self.data = data.type(torch.float)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        return DatasetOutput(data=x)


def main(args):

    ### Load data
    train_data = torch.tensor(
        np.load(os.path.join(PATH, f"data/mnist", "train_data.npz"))["data"] / 255.0
    )
    eval_data = torch.tensor(
        np.load(os.path.join(PATH, f"data/mnist", "eval_data.npz"))["data"] / 255.0
    )

    train_dataset = MNIST(train_data)
    eval_dataset = MNIST(eval_data)

    model_config = VQVAEConfig(
        input_dim=(1, 28, 28), latent_dim=16, use_ema=True, num_embeddings=256
    )

    encoder = Encoder_ResNet_VQVAE_MNIST(model_config)
    decoder = Decoder_ResNet_VQVAE_MNIST(model_config)

    model = VQVAE(model_config=model_config, encoder=encoder, decoder=decoder)

    gpu_ids = os.environ["SLURM_STEP_GPUS"].split(",")

    training_config = BaseTrainerConfig(
        num_epochs=100,
        output_dir="my_models_on_mnist",
        per_device_train_batch_size=256,
        per_device_eval_batch_size=256,
        learning_rate=1e-3,
        steps_saving=None,
        steps_predict=None,
        no_cuda=False,
        world_size=int(os.environ["SLURM_NTASKS"]),
        dist_backend="nccl",
        rank=int(os.environ["SLURM_PROCID"]),
        local_rank=int(os.environ["SLURM_LOCALID"]),
        master_addr=hostlist.expand_hostlist(os.environ["SLURM_JOB_NODELIST"])[0],
        master_port=str(12345 + int(min(gpu_ids))),
    )

    if int(os.environ["SLURM_PROCID"]) == 0:
        logger.info(model)
        logger.info(f"Training config: {training_config}\n")

    callbacks = []

    # Only log to wandb if main process
    if args.use_wandb and (training_config.rank == 0 or training_config == -1):
        from pythae.trainers.training_callbacks import WandbCallback

        wandb_cb = WandbCallback()
        wandb_cb.setup(
            training_config,
            model_config=model_config,
            project_name=args.wandb_project,
            entity_name=args.wandb_entity,
        )

        callbacks.append(wandb_cb)

    trainer = BaseTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        training_config=training_config,
        callbacks=callbacks,
    )

    start_time = time.time()

    trainer.train()

    end_time = time.time()

    logger.info(f"Total execution time: {(end_time - start_time)} seconds")


if __name__ == "__main__":

    main(args)
