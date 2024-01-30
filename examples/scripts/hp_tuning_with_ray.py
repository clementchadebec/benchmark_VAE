from pythae.pipelines import TrainingPipeline
from pythae.models import VAE, VAEConfig
from pythae.trainers import BaseTrainerConfig, BaseTrainer
from pythae.data.datasets import BaseDataset
import torch
import numpy as np

import torchvision.datasets as datasets

from ray import air, tune
from ray.tune.schedulers import ASHAScheduler
from pythae.trainers.training_callbacks import TrainingCallback

class RayCallback(TrainingCallback):

    def __init__(self) -> None:
        super().__init__()

    def on_epoch_end(self, training_config: BaseTrainerConfig, **kwargs):
        metrics = kwargs.pop("metrics")
        tune.report(eval_epoch_loss=metrics["eval_epoch_loss"])
        
def train_ray(config):

    mnist_trainset = datasets.MNIST(root='../../data', train=True, download=True, transform=None)

    train_dataset = BaseDataset(mnist_trainset.data[:1000].reshape(-1, 1, 28, 28) / 255., torch.ones(1000))
    eval_dataset = BaseDataset(mnist_trainset.data[-1000:].reshape(-1, 1, 28, 28) / 255., torch.ones(1000))

    my_training_config = BaseTrainerConfig(
    output_dir='my_model',
    num_epochs=50,
    learning_rate=config["lr"],
    per_device_train_batch_size=200,
    per_device_eval_batch_size=200,
    steps_saving=None,
    optimizer_cls="AdamW",
    optimizer_params={"weight_decay": 0.05, "betas": (0.91, 0.995)},
    scheduler_cls="ReduceLROnPlateau",
    scheduler_params={"patience": 5, "factor": 0.5}
    )

    my_vae_config = model_config = VAEConfig(
    input_dim=(1, 28, 28),
    latent_dim=10
    )

    my_vae_model = VAE(
    model_config=my_vae_config
    )

    callbacks = [RayCallback()]

    trainer = BaseTrainer(my_vae_model, train_dataset, eval_dataset, my_training_config, callbacks=callbacks)

    trainer.train()

    
search_space = {
    "lr": tune.sample_from(lambda spec: 10 ** (-10 * np.random.rand())),
    "momentum": tune.uniform(0.1, 0.9),
}

tuner = tune.Tuner(
    train_ray,
    tune_config=tune.TuneConfig(
        num_samples=20,
        scheduler=ASHAScheduler(metric="eval_epoch_loss", mode="min"),
    ),
    param_space=search_space,
)

results = tuner.fit()