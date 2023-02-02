import argparse
import logging
import os
from statistics import mode

import hostlist
import numpy as np
import torch

from pythae.data.datasets import BaseDataset
from pythae.models import VQVAE, VQVAEConfig
from pythae.trainers import BaseTrainer, BaseTrainerConfig

logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)

PATH = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()

# Training setting
ap.add_argument(
    "--model_config",
    help="path to model config file (expected json file)",
    default=None,
)
ap.add_argument(
    "--training_config",
    help="path to training config_file (expected json file)",
    default=os.path.join(PATH, "configs/base_training_config.json"),
)
ap.add_argument(
    "--use_wandb",
    help="whether to log the metrics in wandb",
    action="store_true",
)
ap.add_argument(
    "--wandb_project",
    help="wandb project name",
    default="test-project",
)
ap.add_argument(
    "--wandb_entity",
    help="wandb entity name",
    default="benchmark_team",
)

args = ap.parse_args()


def main(args):

    train_data = torch.rand(10000, 2)
    eval_data = torch.rand(10000, 2)

    train_dataset = BaseDataset(train_data, labels=torch.ones(10000))
    eval_dataset = BaseDataset(eval_data, labels=torch.ones(10000))

    model_config = VQVAEConfig(input_dim=(2,), latent_dim=2)

    model = VQVAE(model_config=model_config)

    gpu_ids = os.environ["SLURM_STEP_GPUS"].split(",")

    training_config = BaseTrainerConfig(
        num_epochs=10,
        output_dir="my_models_on_mnist",
        learning_rate=1e-3,
        steps_saving=2,
        steps_predict=3,
        no_cuda=False,
        world_size=int(os.environ["SLURM_NTASKS"]),
        rank=int(os.environ["SLURM_PROCID"]),
        local_rank=int(os.environ["SLURM_LOCALID"]),
        master_addr=hostlist.expand_hostlist(os.environ["SLURM_JOB_NODELIST"])[0],
        master_port=str(12345 + int(min(gpu_ids))),
    )

    if int(os.environ["SLURM_PROCID"]) == 0:
        logger.info(model)
        logger.info(f"Training config: {training_config}\n")

    callbacks = []

    if args.use_wandb:
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

    trainer.train()


if __name__ == "__main__":

    main(args)
