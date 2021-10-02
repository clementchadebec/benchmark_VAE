import os
from copy import deepcopy

import pytest
import torch
from torch.optim import SGD, Adadelta, Adagrad, Adam, RMSprop

from pyraug.customexception import ModelError
from pyraug.models import RHVAE, BaseVAE
from pyraug.models.base.base_config import BaseModelConfig
from pyraug.models.rhvae.rhvae_config import RHVAEConfig
from pyraug.trainers.trainers import Trainer
from pyraug.trainers.training_config import TrainingConfig
from tests.data.rhvae.custom_architectures import *

PATH = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def train_dataset():
    return torch.load(os.path.join(PATH, "data/mnist_clean_train_dataset_sample"))


@pytest.fixture()
def model_sample():
    return BaseVAE(BaseModelConfig(input_dim=784))


@pytest.fixture
def training_config(tmpdir):
    tmpdir.mkdir("dummy_folder")
    dir_path = os.path.join(tmpdir, "dummy_folder")
    return TrainingConfig(output_dir=dir_path)


class Test_DataLoader:
    @pytest.fixture(
        params=[
            TrainingConfig(),
            TrainingConfig(batch_size=100),
            TrainingConfig(batch_size=10),
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
        trainer = Trainer(
            model=model_sample,
            train_dataset=train_dataset,
            training_config=training_config_batch_size,
        )

        train_data_loader = trainer.get_train_dataloader(train_dataset)

        assert issubclass(type(train_data_loader), torch.utils.data.DataLoader)
        assert train_data_loader.dataset == train_dataset

        assert train_data_loader.batch_size == trainer.training_config.batch_size

    def test_build_eval_data_loader(
        self, model_sample, train_dataset, training_config_batch_size
    ):
        trainer = Trainer(
            model=model_sample,
            train_dataset=train_dataset,
            training_config=training_config_batch_size,
        )

        train_data_loader = trainer.get_eval_dataloader(train_dataset)

        assert issubclass(type(train_data_loader), torch.utils.data.DataLoader)
        assert train_data_loader.dataset == train_dataset

        assert train_data_loader.batch_size == trainer.training_config.batch_size


class Test_Set_Training_config:
    @pytest.fixture(
        params=[TrainingConfig(), TrainingConfig(batch_size=10, learning_rate=1e-5)]
    )
    def training_configs(self, request, tmpdir):
        tmpdir.mkdir("dummy_folder")
        dir_path = os.path.join(tmpdir, "dummy_folder")
        request.param.output_dir = dir_path
        return request.param

    def test_set_training_config(self, model_sample, train_dataset, training_configs):
        trainer = Trainer(
            model=model_sample,
            train_dataset=train_dataset,
            training_config=training_configs,
        )

        # check if default config is set
        if training_configs is None:
            assert trainer.training_config == TrainingConfig(
                output_dir="dummy_output_dir"
            )

        else:
            assert trainer.training_config == training_configs


class Test_Build_Optimizer:
    @pytest.fixture(params=[TrainingConfig(), TrainingConfig(learning_rate=1e-5)])
    def training_configs_learning_rate(self, tmpdir, request):
        request.param.output_dir = tmpdir.mkdir("dummy_folder")
        return request.param

    @pytest.fixture(params=[Adagrad, Adam, Adadelta, SGD, RMSprop])
    def optimizers(self, request, model_sample, training_configs_learning_rate):

        optimizer = request.param(
            model_sample.parameters(), lr=training_configs_learning_rate.learning_rate
        )
        return optimizer

    def test_default_optimizer_building(
        self, model_sample, train_dataset, training_configs_learning_rate
    ):

        trainer = Trainer(
            model=model_sample,
            train_dataset=train_dataset,
            training_config=training_configs_learning_rate,
            optimizer=None,
        )

        assert issubclass(type(trainer.optimizer), torch.optim.Adam)
        assert (
            trainer.optimizer.defaults["lr"]
            == training_configs_learning_rate.learning_rate
        )

    def test_set_custom_optimizer(
        self, model_sample, train_dataset, training_configs_learning_rate, optimizers
    ):
        trainer = Trainer(
            model=model_sample,
            train_dataset=train_dataset,
            training_config=training_configs_learning_rate,
            optimizer=optimizers,
        )

        assert issubclass(type(trainer.optimizer), type(optimizers))
        assert (
            trainer.optimizer.defaults["lr"]
            == training_configs_learning_rate.learning_rate
        )


class Test_Init_EarlyStopping_Flags:
    @pytest.fixture(
        params=[
            (TrainingConfig(), (True, False)),
            (
                TrainingConfig(train_early_stopping=50, eval_early_stopping=None),
                (True, False),
            ),
            (
                TrainingConfig(train_early_stopping=50, eval_early_stopping=50),
                (False, True),
            ),
            (
                TrainingConfig(train_early_stopping=None, eval_early_stopping=50),
                (False, True),
            ),
            (
                TrainingConfig(train_early_stopping=None, eval_early_stopping=None),
                (False, False),
            ),
        ]
    )
    def training_config_early_stopping(self, request, tmpdir):
        tmpdir.mkdir("dummy_folder")
        dir_path = os.path.join(tmpdir, "dummy_folder")
        request.param[0].output_dir = dir_path
        return request.param

    def test_set_early_stopping_flags(
        self, model_sample, train_dataset, training_config_early_stopping
    ):
        training_config = training_config_early_stopping[0]

        # test with an eval dataset not None
        true_flag_train_es = training_config_early_stopping[1][0]
        true_flag_eval_es = training_config_early_stopping[1][1]

        trainer = Trainer(
            model=model_sample,
            train_dataset=train_dataset,
            eval_dataset=train_dataset,
            training_config=training_config,
        )

        assert trainer.make_train_early_stopping == true_flag_train_es
        assert trainer.make_eval_early_stopping == true_flag_eval_es

        # test with an eval dataset which is None

        true_flag_train_es = False

        if training_config is None or training_config.train_early_stopping is not None:
            true_flag_train_es = True

        true_flag_eval_es = False

        trainer = Trainer(
            model=model_sample,
            train_dataset=train_dataset,
            eval_dataset=None,
            training_config=training_config,
        )

        assert trainer.make_train_early_stopping == true_flag_train_es
        assert trainer.make_eval_early_stopping == true_flag_eval_es


class Test_Device_Checks:
    @pytest.fixture(
        params=[
            TrainingConfig(max_epochs=3, no_cuda=True),
            TrainingConfig(max_epochs=3, no_cuda=False),
        ]
    )
    def training_configs(self, tmpdir, request):
        tmpdir.mkdir("dummy_folder")
        dir_path = os.path.join(tmpdir, "dummy_folder")
        request.param.output_dir = dir_path
        return request.param

    @pytest.fixture(
        params=[RHVAEConfig(input_dim=784), RHVAEConfig(input_dim=784, latent_dim=5)]
    )
    def rhvae_config(self, request):
        return request.param

    @pytest.fixture
    def custom_encoder(self, rhvae_config):
        return Encoder_MLP_Custom(rhvae_config)

    @pytest.fixture
    def custom_decoder(self, rhvae_config):
        return Decoder_MLP_Custom(rhvae_config)

    @pytest.fixture
    def custom_metric(self, rhvae_config):
        return Metric_MLP_Custom(rhvae_config)

    @pytest.fixture(params=[torch.rand(1), torch.rand(1), torch.rand(1)])
    def rhvae_sample(
        self, rhvae_config, custom_encoder, custom_decoder, custom_metric, request
    ):
        # randomized

        alpha = request.param

        if alpha < 0.125:
            model = RHVAE(rhvae_config)

        elif 0.125 <= alpha < 0.25:
            model = RHVAE(rhvae_config, encoder=custom_encoder)

        elif 0.25 <= alpha < 0.375:
            model = RHVAE(rhvae_config, decoder=custom_decoder)

        elif 0.375 <= alpha < 0.5:
            model = RHVAE(rhvae_config, metric=custom_metric)

        elif 0.5 <= alpha < 0.625:
            model = RHVAE(rhvae_config, encoder=custom_encoder, decoder=custom_decoder)

        elif 0.625 <= alpha < 0:
            model = RHVAE(rhvae_config, encoder=custom_encoder, metric=custom_metric)

        elif 0.750 <= alpha < 0.875:
            model = RHVAE(rhvae_config, decoder=custom_decoder, metric=custom_metric)

        else:
            model = RHVAE(
                rhvae_config,
                encoder=custom_encoder,
                decoder=custom_decoder,
                metric=custom_metric,
            )

        return model

    @pytest.fixture(params=[None, Adagrad, Adam, Adadelta, SGD, RMSprop])
    def optimizers(self, request, rhvae_sample, training_configs):
        if request.param is not None:
            optimizer = request.param(
                rhvae_sample.parameters(), lr=training_configs.learning_rate
            )

        else:
            optimizer = None

        return optimizer

    def test_set_on_device(self, rhvae_sample, train_dataset, training_config):
        trainer = Trainer(
            model=rhvae_sample,
            train_dataset=train_dataset,
            training_config=training_config,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"

        if training_config.no_cuda:
            assert next(trainer.model.parameters()).device == "cpu"

        else:
            next(trainer.model.parameters()).device == device


class Test_Sanity_Checks:
    @pytest.fixture
    def rhvae_config(self):
        return RHVAEConfig(input_dim=784)

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
    def rhvae_sample(
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
            rhvae_config.input_dim = (
                rhvae_config.input_dim - 1
            )  # create error on input dim
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

    def test_raises_sanity_check_error(
        self, rhvae_sample, train_dataset, training_config
    ):
        trainer = Trainer(
            model=rhvae_sample,
            train_dataset=train_dataset,
            training_config=training_config,
        )

        with pytest.raises(ModelError):
            trainer._run_model_sanity_check(rhvae_sample, train_dataset)


class Test_RHVAE_Training:
    @pytest.fixture(params=[TrainingConfig(max_epochs=3)])
    def training_configs(self, tmpdir, request):
        tmpdir.mkdir("dummy_folder")
        dir_path = os.path.join(tmpdir, "dummy_folder")
        request.param.output_dir = dir_path
        return request.param

    @pytest.fixture(
        params=[RHVAEConfig(input_dim=784), RHVAEConfig(input_dim=784, latent_dim=5)]
    )
    def rhvae_config(self, request):
        return request.param

    @pytest.fixture
    def custom_encoder(self, rhvae_config):
        return Encoder_MLP_Custom(rhvae_config)

    @pytest.fixture
    def custom_decoder(self, rhvae_config):
        return Decoder_MLP_Custom(rhvae_config)

    @pytest.fixture
    def custom_metric(self, rhvae_config):
        return Metric_MLP_Custom(rhvae_config)

    @pytest.fixture(
        params=[
            torch.rand(1),
            torch.rand(1),
            torch.rand(1),
            torch.rand(1),
            torch.rand(1),
        ]
    )
    def rhvae_sample(
        self, rhvae_config, custom_encoder, custom_decoder, custom_metric, request
    ):
        # randomized

        alpha = request.param

        if alpha < 0.125:
            model = RHVAE(rhvae_config)

        elif 0.125 <= alpha < 0.25:
            model = RHVAE(rhvae_config, encoder=custom_encoder)

        elif 0.25 <= alpha < 0.375:
            model = RHVAE(rhvae_config, decoder=custom_decoder)

        elif 0.375 <= alpha < 0.5:
            model = RHVAE(rhvae_config, metric=custom_metric)

        elif 0.5 <= alpha < 0.625:
            model = RHVAE(rhvae_config, encoder=custom_encoder, decoder=custom_decoder)

        elif 0.625 <= alpha < 0:
            model = RHVAE(rhvae_config, encoder=custom_encoder, metric=custom_metric)

        elif 0.750 <= alpha < 0.875:
            model = RHVAE(rhvae_config, decoder=custom_decoder, metric=custom_metric)

        else:
            model = RHVAE(
                rhvae_config,
                encoder=custom_encoder,
                decoder=custom_decoder,
                metric=custom_metric,
            )

        return model

    @pytest.fixture(params=[None, Adagrad, Adam, Adadelta, SGD, RMSprop])
    def optimizers(self, request, rhvae_sample, training_configs):
        if request.param is not None:
            optimizer = request.param(
                rhvae_sample.parameters(), lr=training_configs.learning_rate
            )

        else:
            optimizer = None

        return optimizer

    def test_rhvae_train_step(
        self, rhvae_sample, train_dataset, training_configs, optimizers
    ):
        trainer = Trainer(
            model=rhvae_sample,
            train_dataset=train_dataset,
            training_config=training_configs,
            optimizer=optimizers,
        )

        start_model_state_dict = deepcopy(trainer.model.state_dict())

        step_1_loss = trainer.train_step()

        step_1_model_state_dict = deepcopy(trainer.model.state_dict())

        # check that weights were updated
        assert not all(
            [
                torch.equal(start_model_state_dict[key], step_1_model_state_dict[key])
                for key in start_model_state_dict.keys()
            ]
        )

    def test_rhvae_eval_step(
        self, rhvae_sample, train_dataset, training_configs, optimizers
    ):
        trainer = Trainer(
            model=rhvae_sample,
            train_dataset=train_dataset,
            eval_dataset=train_dataset,
            training_config=training_configs,
            optimizer=optimizers,
        )

        start_model_state_dict = deepcopy(trainer.model.state_dict())

        step_1_loss = trainer.eval_step()

        step_1_model_state_dict = deepcopy(trainer.model.state_dict())

        # check that weights were updated
        assert all(
            [
                torch.equal(start_model_state_dict[key], step_1_model_state_dict[key])
                for key in start_model_state_dict.keys()
            ]
        )

    def test_rhvae_main_train_loop(
        self, tmpdir, rhvae_sample, train_dataset, training_configs, optimizers
    ):

        trainer = Trainer(
            model=rhvae_sample,
            train_dataset=train_dataset,
            eval_dataset=train_dataset,
            training_config=training_configs,
            optimizer=optimizers,
        )

        start_model_state_dict = deepcopy(trainer.model.state_dict())

        trainer.train()

        step_1_model_state_dict = deepcopy(trainer.model.state_dict())

        # check that weights were updated
        assert not all(
            [
                torch.equal(start_model_state_dict[key], step_1_model_state_dict[key])
                for key in start_model_state_dict.keys()
            ]
        )


class Test_RHVAE_Saving:
    @pytest.fixture(
        params=[
            TrainingConfig(max_epochs=4, steps_saving=3),
            TrainingConfig(max_epochs=3, steps_saving=2, learning_rate=1e-5),
        ]
    )
    def training_configs(self, tmpdir, request):
        tmpdir.mkdir("dummy_folder")
        dir_path = os.path.join(tmpdir, "dummy_folder")
        request.param.output_dir = dir_path
        return request.param

    @pytest.fixture(
        params=[RHVAEConfig(input_dim=784), RHVAEConfig(input_dim=784, latent_dim=5)]
    )
    def rhvae_config(self, request):
        return request.param

    @pytest.fixture
    def custom_encoder(self, rhvae_config):
        return Encoder_MLP_Custom(rhvae_config)

    @pytest.fixture
    def custom_decoder(self, rhvae_config):
        return Decoder_MLP_Custom(rhvae_config)

    @pytest.fixture
    def custom_metric(self, rhvae_config):
        return Metric_MLP_Custom(rhvae_config)

    @pytest.fixture(params=[torch.rand(1), torch.rand(1), torch.rand(1)])
    def rhvae_sample(
        self, rhvae_config, custom_encoder, custom_decoder, custom_metric, request
    ):
        # randomized

        alpha = request.param

        if alpha < 0.125:
            model = RHVAE(rhvae_config)

        elif 0.125 <= alpha < 0.25:
            model = RHVAE(rhvae_config, encoder=custom_encoder)

        elif 0.25 <= alpha < 0.375:
            model = RHVAE(rhvae_config, decoder=custom_decoder)

        elif 0.375 <= alpha < 0.5:
            model = RHVAE(rhvae_config, metric=custom_metric)

        elif 0.5 <= alpha < 0.625:
            model = RHVAE(rhvae_config, encoder=custom_encoder, decoder=custom_decoder)

        elif 0.625 <= alpha < 0:
            model = RHVAE(rhvae_config, encoder=custom_encoder, metric=custom_metric)

        elif 0.750 <= alpha < 0.875:
            model = RHVAE(rhvae_config, decoder=custom_decoder, metric=custom_metric)

        else:
            model = RHVAE(
                rhvae_config,
                encoder=custom_encoder,
                decoder=custom_decoder,
                metric=custom_metric,
            )

        return model

    @pytest.fixture(params=[None, Adagrad, Adam, Adadelta, SGD, RMSprop])
    def optimizers(self, request, rhvae_sample, training_configs):
        if request.param is not None:
            optimizer = request.param(
                rhvae_sample.parameters(), lr=training_configs.learning_rate
            )

        else:
            optimizer = None

        return optimizer

    def test_checkpoint_saving(
        self, tmpdir, rhvae_sample, train_dataset, training_configs, optimizers
    ):

        dir_path = training_configs.output_dir

        trainer = Trainer(
            model=rhvae_sample,
            train_dataset=train_dataset,
            training_config=training_configs,
            optimizer=optimizers,
        )

        # Make a training step
        step_1_loss = trainer.train_step()

        model = deepcopy(trainer.model)
        optimizer = deepcopy(trainer.optimizer)

        trainer.save_checkpoint(dir_path=dir_path, epoch=0)

        checkpoint_dir = os.path.join(dir_path, "checkpoint_epoch_0")

        assert os.path.isdir(checkpoint_dir)

        files_list = os.listdir(checkpoint_dir)

        assert set(["model.pt", "optimizer.pt", "training_config.json"]).issubset(
            set(files_list)
        )

        # check pickled custom decoder
        if not rhvae_sample.model_config.uses_default_decoder:
            assert "decoder.pkl" in files_list

        else:
            assert not "decoder.pkl" in files_list

        # check pickled custom encoder
        if not rhvae_sample.model_config.uses_default_encoder:
            assert "encoder.pkl" in files_list

        else:
            assert not "encoder.pkl" in files_list

        # check pickled custom metric
        if not rhvae_sample.model_config.uses_default_metric:
            assert "metric.pkl" in files_list

        else:
            assert not "metric.pkl" in files_list

        model_rec_state_dict = torch.load(os.path.join(checkpoint_dir, "model.pt"))[
            "model_state_dict"
        ]

        assert all(
            [
                torch.equal(
                    model_rec_state_dict[key].cpu(), model.state_dict()[key].cpu()
                )
                for key in model.state_dict().keys()
            ]
        )

        # check reload full model
        model_rec = RHVAE.load_from_folder(os.path.join(checkpoint_dir))

        assert all(
            [
                torch.equal(
                    model_rec.state_dict()[key].cpu(), model.state_dict()[key].cpu()
                )
                for key in model.state_dict().keys()
            ]
        )

        assert torch.equal(model_rec.M_tens.cpu(), model.M_tens.cpu())
        assert torch.equal(model_rec.centroids_tens.cpu(), model.centroids_tens.cpu())
        assert type(model_rec.encoder.cpu()) == type(model.encoder.cpu())
        assert type(model_rec.decoder.cpu()) == type(model.decoder.cpu())
        assert type(model_rec.metric.cpu()) == type(model.metric.cpu())

        optim_rec_state_dict = torch.load(os.path.join(checkpoint_dir, "optimizer.pt"))

        assert all(
            [
                dict_rec == dict_optimizer
                for (dict_rec, dict_optimizer) in zip(
                    optim_rec_state_dict["param_groups"],
                    optimizer.state_dict()["param_groups"],
                )
            ]
        )

        assert all(
            [
                dict_rec == dict_optimizer
                for (dict_rec, dict_optimizer) in zip(
                    optim_rec_state_dict["state"], optimizer.state_dict()["state"]
                )
            ]
        )

    def test_checkpoint_saving_during_training(
        self, tmpdir, rhvae_sample, train_dataset, training_configs, optimizers
    ):
        #
        target_saving_epoch = training_configs.steps_saving

        dir_path = training_configs.output_dir

        trainer = Trainer(
            model=rhvae_sample,
            train_dataset=train_dataset,
            training_config=training_configs,
            optimizer=optimizers,
        )

        model = deepcopy(trainer.model)

        trainer.train()

        training_dir = os.path.join(dir_path, f"training_{trainer._training_signature}")
        assert os.path.isdir(training_dir)

        checkpoint_dir = os.path.join(
            training_dir, f"checkpoint_epoch_{target_saving_epoch}"
        )

        assert os.path.isdir(checkpoint_dir)

        files_list = os.listdir(checkpoint_dir)

        # check files
        assert set(["model.pt", "optimizer.pt", "training_config.json"]).issubset(
            set(files_list)
        )

        # check pickled custom decoder
        if not rhvae_sample.model_config.uses_default_decoder:
            assert "decoder.pkl" in files_list

        else:
            assert not "decoder.pkl" in files_list

        # check pickled custom encoder
        if not rhvae_sample.model_config.uses_default_encoder:
            assert "encoder.pkl" in files_list

        else:
            assert not "encoder.pkl" in files_list

        # check pickled custom metric
        if not rhvae_sample.model_config.uses_default_metric:
            assert "metric.pkl" in files_list

        else:
            assert not "metric.pkl" in files_list

        model_rec_state_dict = torch.load(os.path.join(checkpoint_dir, "model.pt"))[
            "model_state_dict"
        ]

        assert not all(
            [
                torch.equal(model_rec_state_dict[key], model.state_dict()[key])
                for key in model.state_dict().keys()
            ]
        )

    def test_final_model_saving(
        self, tmpdir, rhvae_sample, train_dataset, training_configs, optimizers
    ):

        dir_path = training_configs.output_dir

        trainer = Trainer(
            model=rhvae_sample,
            train_dataset=train_dataset,
            training_config=training_configs,
            optimizer=optimizers,
        )

        trainer.train()

        model = deepcopy(trainer.model)

        training_dir = os.path.join(dir_path, f"training_{trainer._training_signature}")
        assert os.path.isdir(training_dir)

        final_dir = os.path.join(training_dir, f"final_model")
        assert os.path.isdir(final_dir)

        files_list = os.listdir(final_dir)

        assert set(["model.pt", "model_config.json", "training_config.json"]).issubset(
            set(files_list)
        )

        # check pickled custom decoder
        if not rhvae_sample.model_config.uses_default_decoder:
            assert "decoder.pkl" in files_list

        else:
            assert not "decoder.pkl" in files_list

        # check pickled custom encoder
        if not rhvae_sample.model_config.uses_default_encoder:
            assert "encoder.pkl" in files_list

        else:
            assert not "encoder.pkl" in files_list

        # check pickled custom metric
        if not rhvae_sample.model_config.uses_default_metric:
            assert "metric.pkl" in files_list

        else:
            assert not "metric.pkl" in files_list

        # check reload full model
        model_rec = RHVAE.load_from_folder(os.path.join(final_dir))

        assert all(
            [
                torch.equal(
                    model_rec.state_dict()[key].cpu(), model.state_dict()[key].cpu()
                )
                for key in model.state_dict().keys()
            ]
        )

        assert torch.equal(model_rec.M_tens.cpu(), model.M_tens.cpu())
        assert torch.equal(model_rec.centroids_tens.cpu(), model.centroids_tens.cpu())
        assert type(model_rec.encoder.cpu()) == type(model.encoder.cpu())
        assert type(model_rec.decoder.cpu()) == type(model.decoder.cpu())
        assert type(model_rec.metric.cpu()) == type(model.metric.cpu())


class Test_Logging:
    @pytest.fixture
    def training_config(self, tmpdir):
        tmpdir.mkdir("dummy_folder")
        dir_path = os.path.join(tmpdir, "dummy_folder")
        return TrainingConfig(output_dir=dir_path, max_epochs=2)

    @pytest.fixture
    def model_sample(self):
        return RHVAE(RHVAEConfig(input_dim=784))

    def test_create_log_file(
        self, tmpdir, model_sample, train_dataset, training_config
    ):
        dir_log_path = os.path.join(tmpdir, "dummy_folder")

        trainer = Trainer(
            model=model_sample,
            train_dataset=train_dataset,
            training_config=training_config,
        )

        trainer.train(log_output_dir=dir_log_path)

        assert os.path.isdir(dir_log_path)
        assert f"training_logs_{trainer._training_signature}.log" in os.listdir(
            dir_log_path
        )
