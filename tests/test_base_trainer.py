import os
from copy import deepcopy

import pytest
import torch

from pythae.customexception import ModelError
from pythae.models import AE, RHVAE, VAE, AEConfig, RHVAEConfig, VAEConfig
from pythae.trainers import BaseTrainer, BaseTrainerConfig
from tests.data.custom_architectures import *

PATH = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def train_dataset():
    return torch.load(os.path.join(PATH, "data/mnist_clean_train_dataset_sample"))


@pytest.fixture()
def model_sample():
    return AE(AEConfig(input_dim=(1, 28, 28)))


@pytest.fixture
def training_config(tmpdir):
    tmpdir.mkdir("dummy_folder")
    dir_path = os.path.join(tmpdir, "dummy_folder")
    return BaseTrainerConfig(output_dir=dir_path)


class Test_DataLoader:
    @pytest.fixture(
        params=[
            BaseTrainerConfig(),
            BaseTrainerConfig(
                per_device_train_batch_size=35, per_device_eval_batch_size=100
            ),
            BaseTrainerConfig(
                per_device_train_batch_size=3, per_device_eval_batch_size=10
            ),
        ]
    )
    def training_config_batch_size(self, request, tmpdir):
        tmpdir.mkdir("dummy_folder")
        dir_path = os.path.join(tmpdir, "dummy_folder")
        request.param.output_dir = dir_path  # this avoids creating a permanent folder
        return request.param

    def test_build_train_data_loader(
        self, model_sample, train_dataset, training_config_batch_size
    ):
        trainer = BaseTrainer(
            model=model_sample,
            train_dataset=train_dataset,
            training_config=training_config_batch_size,
        )

        train_data_loader = trainer.get_train_dataloader(train_dataset)

        assert issubclass(type(train_data_loader), torch.utils.data.DataLoader)
        assert train_data_loader.dataset == train_dataset

        assert (
            train_data_loader.batch_size
            == trainer.training_config.per_device_train_batch_size
        )

    def test_build_eval_data_loader(
        self, model_sample, train_dataset, training_config_batch_size
    ):
        trainer = BaseTrainer(
            model=model_sample,
            train_dataset=train_dataset,
            training_config=training_config_batch_size,
        )

        eval_data_loader = trainer.get_eval_dataloader(train_dataset)

        assert issubclass(type(eval_data_loader), torch.utils.data.DataLoader)
        assert eval_data_loader.dataset == train_dataset

        assert (
            eval_data_loader.batch_size
            == trainer.training_config.per_device_eval_batch_size
        )


class Test_Set_Training_config:
    @pytest.fixture(
        params=[
            None,
            BaseTrainerConfig(),
            BaseTrainerConfig(
                per_device_train_batch_size=10,
                per_device_eval_batch_size=20,
                learning_rate=1e-5,
                optimizer_cls="AdamW",
                optimizer_params={"weight_decay": 0.01},
                scheduler_cls="ExponentialLR",
                scheduler_params={"gamma": 0.321},
            ),
        ]
    )
    def training_configs(self, request, tmpdir):
        if request.param is not None:
            tmpdir.mkdir("dummy_folder")
            dir_path = os.path.join(tmpdir, "dummy_folder")
            request.param.output_dir = dir_path
            return request.param
        else:
            return None

    def test_set_training_config(self, model_sample, train_dataset, training_configs):
        trainer = BaseTrainer(
            model=model_sample,
            train_dataset=train_dataset,
            training_config=training_configs,
        )

        # check if default config is set
        if training_configs is None:
            assert trainer.training_config == BaseTrainerConfig(
                output_dir="dummy_output_dir", keep_best_on_train=True
            )

        else:
            assert trainer.training_config == training_configs


class Test_Build_Optimizer:
    def test_wrong_optimizer_cls(self):
        with pytest.raises(AttributeError):
            BaseTrainerConfig(optimizer_cls="WrongOptim")

    def test_wrong_optimizer_params(self):
        with pytest.raises(TypeError):
            BaseTrainerConfig(
                optimizer_cls="Adam", optimizer_params={"wrong_config": 1}
            )

    @pytest.fixture(params=[BaseTrainerConfig(), BaseTrainerConfig(learning_rate=1e-5)])
    def training_configs_learning_rate(self, tmpdir, request):
        request.param.output_dir = tmpdir.mkdir("dummy_folder")
        return request.param

    @pytest.fixture(
        params=[
            {"optimizer_cls": "Adagrad", "optimizer_params": {"lr_decay": 0.1}},
            {"optimizer_cls": "AdamW", "optimizer_params": {"betas": (0.1234, 0.4321)}},
            {"optimizer_cls": "SGD", "optimizer_params": None},
        ]
    )
    def optimizer_config(self, request, training_configs_learning_rate):

        optimizer_config = request.param

        # set optim and params to training config
        training_configs_learning_rate.optimizer_cls = optimizer_config["optimizer_cls"]
        training_configs_learning_rate.optimizer_params = optimizer_config[
            "optimizer_params"
        ]

        return optimizer_config

    def test_default_optimizer_building(
        self, model_sample, train_dataset, training_configs_learning_rate
    ):

        trainer = BaseTrainer(
            model=model_sample,
            train_dataset=train_dataset,
            training_config=training_configs_learning_rate,
        )

        trainer.set_optimizer()

        assert issubclass(type(trainer.optimizer), torch.optim.Adam)
        assert (
            trainer.optimizer.defaults["lr"]
            == training_configs_learning_rate.learning_rate
        )

    def test_set_custom_optimizer(
        self,
        model_sample,
        train_dataset,
        training_configs_learning_rate,
        optimizer_config,
    ):

        trainer = BaseTrainer(
            model=model_sample,
            train_dataset=train_dataset,
            training_config=training_configs_learning_rate,
        )

        trainer.set_optimizer()

        assert issubclass(
            type(trainer.optimizer),
            getattr(torch.optim, optimizer_config["optimizer_cls"]),
        )
        assert (
            trainer.optimizer.defaults["lr"]
            == training_configs_learning_rate.learning_rate
        )
        if optimizer_config["optimizer_params"] is not None:
            assert all(
                [
                    trainer.optimizer.defaults[key]
                    == optimizer_config["optimizer_params"][key]
                    for key in optimizer_config["optimizer_params"].keys()
                ]
            )


class Test_Build_Scheduler:
    def test_wrong_scheduler_cls(self):
        with pytest.raises(AttributeError):
            BaseTrainerConfig(scheduler_cls="WrongOptim")

    def test_wrong_scheduler_params(self):
        with pytest.raises(TypeError):
            BaseTrainerConfig(
                scheduler_cls="ReduceLROnPlateau", scheduler_params={"wrong_config": 1}
            )

    @pytest.fixture(params=[BaseTrainerConfig(), BaseTrainerConfig(learning_rate=1e-5)])
    def training_configs_learning_rate(self, tmpdir, request):
        request.param.output_dir = tmpdir.mkdir("dummy_folder")
        return request.param

    @pytest.fixture(
        params=[
            {"optimizer_cls": "Adagrad", "optimizer_params": {"lr_decay": 0.1}},
            {"optimizer_cls": "AdamW", "optimizer_params": {"betas": (0.1234, 0.4321)}},
            {"optimizer_cls": "SGD", "optimizer_params": None},
        ]
    )
    def optimizer_config(self, request, training_configs_learning_rate):

        optimizer_config = request.param

        # set optim and params to training config
        training_configs_learning_rate.optimizer_cls = optimizer_config["optimizer_cls"]
        training_configs_learning_rate.optimizer_params = optimizer_config[
            "optimizer_params"
        ]

        return optimizer_config

    @pytest.fixture(
        params=[
            {"scheduler_cls": "StepLR", "scheduler_params": {"step_size": 1}},
            {"scheduler_cls": "LinearLR", "scheduler_params": None},
            {"scheduler_cls": "ExponentialLR", "scheduler_params": {"gamma": 3.14}},
        ]
    )
    def scheduler_config(self, request, training_configs_learning_rate):

        scheduler_config = request.param

        # set scheduler and params to training config
        training_configs_learning_rate.scheduler_cls = scheduler_config["scheduler_cls"]
        training_configs_learning_rate.scheduler_params = scheduler_config[
            "scheduler_params"
        ]

        return request.param

    def test_default_scheduler_building(
        self, model_sample, train_dataset, training_configs_learning_rate
    ):

        trainer = BaseTrainer(
            model=model_sample,
            train_dataset=train_dataset,
            training_config=training_configs_learning_rate,
        )

        trainer.set_optimizer()
        trainer.set_scheduler()

        assert trainer.scheduler is None

    def test_set_custom_scheduler(
        self,
        model_sample,
        train_dataset,
        training_configs_learning_rate,
        scheduler_config,
    ):
        trainer = BaseTrainer(
            model=model_sample,
            train_dataset=train_dataset,
            training_config=training_configs_learning_rate,
        )

        trainer.set_optimizer()
        trainer.set_scheduler()

        assert issubclass(
            type(trainer.scheduler),
            getattr(torch.optim.lr_scheduler, scheduler_config["scheduler_cls"]),
        )
        if scheduler_config["scheduler_params"] is not None:
            assert all(
                [
                    trainer.scheduler.state_dict()[key]
                    == scheduler_config["scheduler_params"][key]
                    for key in scheduler_config["scheduler_params"].keys()
                ]
            )


class Test_Device_Checks:
    def test_set_environ_variable(self):
        os.environ["LOCAL_RANK"] = "1"
        os.environ["WORLD_SIZE"] = "4"
        os.environ["RANK"] = "3"
        os.environ["MASTER_ADDR"] = "314"
        os.environ["MASTER_PORT"] = "222"

        trainer_config = BaseTrainerConfig()

        assert int(trainer_config.local_rank) == 1
        assert int(trainer_config.world_size) == 4
        assert int(trainer_config.rank) == 3
        assert trainer_config.master_addr == "314"
        assert trainer_config.master_port == "222"

        del os.environ["LOCAL_RANK"]
        del os.environ["WORLD_SIZE"]
        del os.environ["RANK"]
        del os.environ["MASTER_ADDR"]
        del os.environ["MASTER_PORT"]

    @pytest.fixture(
        params=[
            BaseTrainerConfig(num_epochs=3, no_cuda=True),
            BaseTrainerConfig(num_epochs=3, no_cuda=False),
        ]
    )
    def training_configs(self, tmpdir, request):
        tmpdir.mkdir("dummy_folder")
        dir_path = os.path.join(tmpdir, "dummy_folder")
        request.param.output_dir = dir_path
        return request.param

    @pytest.fixture(
        params=[
            AEConfig(input_dim=(1, 28, 28)),
            AEConfig(input_dim=(1, 28, 28), latent_dim=5),
        ]
    )
    def ae_config(self, request):
        return request.param

    @pytest.fixture
    def custom_encoder(self, ae_config):
        return Encoder_AE_MLP_Custom(ae_config)

    @pytest.fixture
    def custom_decoder(self, ae_config):
        return Decoder_MLP_Custom(ae_config)

    @pytest.fixture(params=[torch.rand(1), torch.rand(1), torch.rand(1)])
    def ae(self, ae_config, custom_encoder, custom_decoder, request):
        # randomized

        alpha = request.param

        if alpha < 0.25:
            model = AE(ae_config)

        elif 0.25 <= alpha < 0.5:
            model = AE(ae_config, encoder=custom_encoder)

        elif 0.5 <= alpha < 0.75:
            model = AE(ae_config, decoder=custom_decoder)

        else:
            model = AE(ae_config, encoder=custom_encoder, decoder=custom_decoder)

        return model

    def test_set_on_device(self, ae, train_dataset, training_config):
        trainer = BaseTrainer(
            model=ae, train_dataset=train_dataset, training_config=training_config
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"

        if training_config.no_cuda:
            assert next(trainer.model.parameters()).device == "cpu"

        else:
            next(trainer.model.parameters()).device == device


class Test_Sanity_Checks:
    @pytest.fixture
    def rhvae_config(self):
        return RHVAEConfig(input_dim=(1, 28, 28))

    @pytest.fixture(
        params=[DecoderWrongInputDim, DecoderWrongOutput, DecoderWrongOutputDim]
    )
    def corrupted_decoder(self, rhvae_config, request):
        return request.param(rhvae_config)

    @pytest.fixture(
        params=[EncoderWrongInputDim, EncoderWrongOutput, EncoderWrongOutputDim]
    )
    def corrupted_encoder(self, rhvae_config, request):
        return request.param(rhvae_config)

    @pytest.fixture(
        params=[
            MetricWrongInputDim,
            MetricWrongOutput,
            MetricWrongOutputDim,
            MetricWrongOutputDimBis,
        ]
    )
    def corrupted_metric(self, rhvae_config, request):
        return request.param(rhvae_config)

    @pytest.fixture(params=[torch.rand(1), torch.rand(1), torch.rand(1)])
    def rhvae(
        self,
        rhvae_config,
        corrupted_encoder,
        corrupted_decoder,
        corrupted_metric,
        request,
    ):
        # randomized

        alpha = request.param

        if alpha < 0.125:
            rhvae_config.input_dim = rhvae_config.input_dim[:-1] + (
                rhvae_config.input_dim[-1] - 1,
            )
            # create error on input dim
            model = RHVAE(rhvae_config)

        elif 0.125 <= alpha < 0.25:
            model = RHVAE(rhvae_config, encoder=corrupted_encoder)

        elif 0.25 <= alpha < 0.375:
            model = RHVAE(rhvae_config, decoder=corrupted_decoder)

        elif 0.375 <= alpha < 0.5:
            model = RHVAE(rhvae_config, metric=corrupted_metric)

        elif 0.5 <= alpha < 0.625:
            model = RHVAE(
                rhvae_config, encoder=corrupted_encoder, decoder=corrupted_decoder
            )

        elif 0.625 <= alpha < 0:
            model = RHVAE(
                rhvae_config, encoder=corrupted_encoder, metric=corrupted_metric
            )

        elif 0.750 <= alpha < 0.875:
            model = RHVAE(
                rhvae_config, decoder=corrupted_decoder, metric=corrupted_metric
            )

        else:
            model = RHVAE(
                rhvae_config,
                encoder=corrupted_encoder,
                decoder=corrupted_decoder,
                metric=corrupted_metric,
            )

        return model

    def test_raises_sanity_check_error(self, rhvae, train_dataset, training_config):
        with pytest.raises(ModelError):
            _ = BaseTrainer(
                model=rhvae,
                train_dataset=train_dataset,
                training_config=training_config,
            )


@pytest.mark.slow
class Test_Main_Training:
    @pytest.fixture(
        params=[
            BaseTrainerConfig(
                num_epochs=3, steps_saving=2, steps_predict=2, learning_rate=1e-5
            )
        ]
    )
    def training_configs(self, tmpdir, request):
        tmpdir.mkdir("dummy_folder")
        dir_path = os.path.join(tmpdir, "dummy_folder")
        request.param.output_dir = dir_path
        return request.param

    @pytest.fixture(
        params=[
            AEConfig(input_dim=(1, 28, 28)),
            VAEConfig(input_dim=(1, 28, 28), latent_dim=5),
        ]
    )
    def ae_config(self, request):
        return request.param

    @pytest.fixture
    def custom_encoder(self, ae_config):
        if isinstance(ae_config, VAEConfig):
            return Encoder_VAE_MLP_Custom(ae_config)
        return Encoder_AE_MLP_Custom(ae_config)

    @pytest.fixture
    def custom_decoder(self, ae_config):
        return Decoder_MLP_Custom(ae_config)

    @pytest.fixture(
        params=[
            torch.rand(1),
            torch.rand(1),
            torch.rand(1),
            torch.rand(1),
            torch.rand(1),
        ]
    )
    def ae(self, ae_config, custom_encoder, custom_decoder, request):
        # randomized

        alpha = request.param

        if alpha < 0.25:
            if isinstance(ae_config, VAEConfig):
                model = VAE(ae_config)
            else:
                model = AE(ae_config)

        elif 0.25 <= alpha < 0.5:
            if isinstance(ae_config, VAEConfig):
                model = VAE(ae_config, encoder=custom_encoder)
            else:
                model = AE(ae_config, encoder=custom_encoder)

        elif 0.5 <= alpha < 0.75:
            if isinstance(ae_config, VAEConfig):
                model = VAE(ae_config, decoder=custom_decoder)
            else:
                model = AE(ae_config, decoder=custom_decoder)

        else:
            if isinstance(ae_config, VAEConfig):
                model = VAE(ae_config, encoder=custom_encoder, decoder=custom_decoder)
            else:
                model = AE(ae_config, encoder=custom_encoder, decoder=custom_decoder)

        return model

    @pytest.fixture(
        params=[
            {"optimizer_cls": "Adagrad", "optimizer_params": {"lr_decay": 0.1}},
            {"optimizer_cls": "AdamW", "optimizer_params": {"betas": (0.1234, 0.4321)}},
            {"optimizer_cls": "SGD", "optimizer_params": None},
        ]
    )
    def optimizer_config(self, request):
        return request.param

    @pytest.fixture(
        params=[
            {"scheduler_cls": "StepLR", "scheduler_params": {"step_size": 0.1}},
            {"scheduler_cls": "ReduceLROnPlateau", "scheduler_params": None},
            {"scheduler_cls": None, "scheduler_params": None},
        ]
    )
    def scheduler_config(self, request):
        return request.param

    @pytest.fixture
    def trainer(
        self, ae, train_dataset, optimizer_config, scheduler_config, training_configs
    ):

        training_configs.optimizer_cls = optimizer_config["optimizer_cls"]
        training_configs.optimizer_params = optimizer_config["optimizer_params"]
        training_configs.scheduler_cls = scheduler_config["scheduler_cls"]
        training_configs.scheduler_params = scheduler_config["scheduler_params"]

        trainer = BaseTrainer(
            model=ae,
            train_dataset=train_dataset,
            eval_dataset=train_dataset,
            training_config=training_configs,
        )

        trainer.prepare_training()

        return trainer

    def test_train_step(self, trainer):

        start_model_state_dict = deepcopy(trainer.model.state_dict())

        step_1_loss = trainer.train_step(epoch=1)

        step_1_model_state_dict = deepcopy(trainer.model.state_dict())

        # check that weights were updated
        for key in start_model_state_dict.keys():
            if "encoder" in key:
                assert not torch.equal(
                    step_1_model_state_dict[key], start_model_state_dict[key]
                ), key

            if "decoder" in key:
                assert not torch.equal(
                    step_1_model_state_dict[key], start_model_state_dict[key]
                )

    def test_eval_step(self, trainer):

        start_model_state_dict = deepcopy(trainer.model.state_dict())

        step_1_loss = trainer.eval_step(epoch=1)

        step_1_model_state_dict = deepcopy(trainer.model.state_dict())

        # check that weights were not updated
        assert all(
            [
                torch.equal(start_model_state_dict[key], step_1_model_state_dict[key])
                for key in start_model_state_dict.keys()
            ]
        )

    def test_predict_step(self, trainer):

        _ = deepcopy(trainer.model.state_dict())

        true_data, recon, gene = trainer.predict(trainer.model)

        assert true_data.reshape(3, -1).shape == recon.reshape(3, -1).shape
        assert gene.reshape(3, -1).shape[1:] == true_data.reshape(3, -1).shape[1:]

    def test_main_train_loop(self, trainer, training_configs):

        start_model_state_dict = deepcopy(trainer.model.state_dict())

        trainer.train()

        step_1_model_state_dict = deepcopy(trainer.model.state_dict())

        # check that weights were updated
        for key in start_model_state_dict.keys():
            if "encoder" in key:
                assert not torch.equal(
                    step_1_model_state_dict[key], start_model_state_dict[key]
                )

            if "decoder" in key:
                assert not torch.equal(
                    step_1_model_state_dict[key], start_model_state_dict[key]
                )

        # check changed lr with custom schedulers
        if (
            type(trainer.scheduler) != torch.optim.lr_scheduler.ReduceLROnPlateau
            and trainer.scheduler is not None
        ):
            assert training_configs.learning_rate != trainer.scheduler.get_last_lr()


class Test_Logging:
    @pytest.fixture
    def training_config(self, tmpdir):
        tmpdir.mkdir("dummy_folder")
        dir_path = os.path.join(tmpdir, "dummy_folder")
        return BaseTrainerConfig(output_dir=dir_path, num_epochs=2)

    @pytest.fixture
    def model_sample(self):
        return RHVAE(RHVAEConfig(input_dim=(1, 28, 28)))

    def test_create_log_file(
        self, tmpdir, model_sample, train_dataset, training_config
    ):
        dir_log_path = os.path.join(tmpdir, "dummy_folder")

        trainer = BaseTrainer(
            model=model_sample,
            train_dataset=train_dataset,
            training_config=training_config,
        )

        trainer.train(log_output_dir=dir_log_path)

        assert os.path.isdir(dir_log_path)
        assert f"training_logs_{trainer._training_signature}.log" in os.listdir(
            dir_log_path
        )
