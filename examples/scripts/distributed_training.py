import argparse
import logging
import os
from statistics import mode
import torch


import numpy as np

from pythae.trainers import BaseTrainerConfig, BaseTrainer
from pythae.models import VQVAE, VQVAEConfig
from pythae.data.datasets import BaseDataset


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
    
    #try:
    #    logger.info(f"\nLoading celeba data...\n")
    #    train_data = (
    #        np.load(os.path.join(PATH, f"data/celeba", "train_data.npz"))[
    #            "data"
    #        ]
    #        / 255.0
    #    )
    #    eval_data = (
    #        np.load(os.path.join(PATH, f"data/celeba", "eval_data.npz"))["data"]
    #        / 255.0
    #    )
    #except Exception as e:
    #    raise FileNotFoundError(
    #        f"Unable to load the data from 'data/{args.dataset}' folder. Please check that both a "
    #        "'train_data.npz' and 'eval_data.npz' are present in the folder.\n Data must be "
    #        " under the key 'data', in the range [0-255] and shaped with channel in first "
    #        "position\n"
    #        f"Exception raised: {type(e)} with message: " + str(e)
    #    ) from e

    #logger.info("Successfully loaded data !\n")
    #logger.info("------------------------------------------------------------")
    #logger.info("Dataset \t \t Shape \t \t \t Range")
    #logger.info(
    #    f"{args.dataset.upper()} train data: \t {train_data.shape} \t [{train_data.min()}-{train_data.max()}] "
    #)
    #logger.info(
    #    f"{args.dataset.upper()} eval data: \t {eval_data.shape} \t [{eval_data.min()}-{eval_data.max()}]"
    #)
    #logger.info("------------------------------------------------------------\n")
#
    #data_input_dim = tuple(train_data.shape[1:])

  
    model_config = VQVAEConfig(
        input_dim=(2,),
        latent_dim=2
    )

    model = VQVAE(
        model_config=model_config
    )

    gpu_ids = os.environ['SLURM_STEP_GPUS'].split(",")

    training_config = BaseTrainerConfig(
        num_epochs=10,
        output_dir="my_models_on_mnist",
        learning_rate=1e-3,
        steps_saving=2,
        steps_predict=3,
        no_cuda=False,
        world_size = int(os.environ['SLURM_NTASKS']),
        rank = int(os.environ['SLURM_PROCID']),
        local_rank = int(os.environ['SLURM_LOCALID']),
        master_addr = os.environ['SLURM_JOB_NODELIST'],
        master_port = str(12345 + int(min(gpu_ids)))
    )

    if int(os.environ['SLURM_PROCID']) == 0:
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
            entity_name=args.wandb_entity
        )

        callbacks.append(wandb_cb)

    trainer = BaseTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        training_config=training_config,
        callbacks=callbacks
    )

    trainer.train()


if __name__ == "__main__":

    main(args)
