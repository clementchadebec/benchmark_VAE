import pytest
import os
import torch

from copy import deepcopy
from torch.optim import Adam

import numpy as np
import shutil

from pythae.models.base.base_utils import ModelOutput
from pythae.models.normalizing_flows import MADE, MADEConfig
from pythae.models.normalizing_flows import NFModel

from pythae.trainers import BaseTrainer, BaseTrainerConfig
from pythae.pipelines import TrainingPipeline

PATH = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(params=[MADEConfig(output_dim=(10, 4)), MADEConfig(input_dim=(5,))])
def model_configs_no_input_output_dim(request):
    return request.param


@pytest.fixture(
    params=[
        MADEConfig(input_dim=(1, 8, 2), output_dim=(1, 10), degrees_ordering='random'),
        MADEConfig(input_dim=(1, 2, 18), output_dim=(1, 5), hidden_sizes=[3, 5, 6]),
    ]
)
def model_configs(request):
    return request.param


class Test_Model_Building:
    def test_build_model(self, model_configs):
        model = MADE(model_configs)
        assert all(
            [
                model.input_dim == np.prod(model_configs.input_dim),
                model.output_dim == np.prod(model_configs.output_dim),
                model.hidden_sizes == model_configs.hidden_sizes,
            ]
        )

    def test_raises_no_input_output_dim(
        self, model_configs_no_input_output_dim):
        with pytest.raises(AttributeError):
            model = MADE(model_configs_no_input_output_dim)


class Test_Model_Saving:

    def test_creates_saving_path(self, model_configs):
        dir_path = os.path.join(PATH, 'test/for/saving')
        model = MADE(model_configs)
        model.save(dir_path=dir_path)
        shutil.rmtree(dir_path)

        dir_path = None
        model = MADE(model_configs)
        with pytest.raises(TypeError) or pytest.raises(FileNotFoundError):
            model.save(dir_path=dir_path)

    def test_default_model_saving(self, tmpdir, model_configs):

        tmpdir.mkdir("dummy_folder")
        dir_path = dir_path = os.path.join(tmpdir, "dummy_folder")

        model = MADE(model_configs)

        model.state_dict()["net.0.weight"][0] = 0

        model.save(dir_path=dir_path)

        assert set(os.listdir(dir_path)) == set(["model_config.json", "model.pt"])

        # reload model
        model_rec = MADE.load_from_folder(dir_path)

        # check configs are the same
        assert model_rec.model_config.__dict__ == model.model_config.__dict__

        assert all(
            [
                torch.equal(model_rec.state_dict()[key], model.state_dict()[key])
                for key in model.state_dict().keys()
            ]
        )

    def test_raises_missing_files(
        self, tmpdir, model_configs):

        tmpdir.mkdir("dummy_folder")
        dir_path = dir_path = os.path.join(tmpdir, "dummy_folder")

        model = MADE(model_configs,)

        model.state_dict()["net.0.weight"][0] = 0

        model.save(dir_path=dir_path)

        os.remove(os.path.join(dir_path, "model.pt"))

        # check raises model.pt is missing
        with pytest.raises(FileNotFoundError):
            model_rec = MADE.load_from_folder(dir_path)

        torch.save({"wrong_key": 0.}, os.path.join(dir_path, "model.pt"))
        # check raises wrong key in model.pt
        with pytest.raises(KeyError):
            model_rec = MADE.load_from_folder(dir_path)

        os.remove(os.path.join(dir_path, "model_config.json"))

        # check raises model_config.json is missing
        with pytest.raises(FileNotFoundError):
            model_rec = MADE.load_from_folder(dir_path)

class Test_Model_forward:
    @pytest.fixture
    def demo_data(self):
        data = torch.load(os.path.join(PATH, "data/mnist_clean_train_dataset_sample"))[
            :
        ]
        return data  # This is an extract of 3 data from MNIST (unnormalized) used to test custom architecture

    @pytest.fixture
    def made(self, model_configs, demo_data):
        model_configs.input_dim = tuple(demo_data["data"][0].shape)
        model_configs.output_dim = tuple(demo_data["data"][0].shape)
        return MADE(model_configs)

    def test_model_train_output(self, made, demo_data):

        made.train()
        out = made(demo_data['data'])

        assert isinstance(out, ModelOutput)

        assert set(["out", "log_abs_det_jac"]) == set(out.keys())

        assert out.out.shape[0] == demo_data["data"].shape[0]
        assert out.log_abs_det_jac.shape[0] == demo_data["data"].shape[0]
        assert out.out.shape[1:] == np.prod(made.model_config.output_dim)
        assert out.log_abs_det_jac.shape[1:] == np.prod(made.model_config.output_dim)

        out = made.inverse(out.out)

        assert isinstance(out, ModelOutput)

        assert set(["out", "log_abs_det_jac"]) == set(out.keys())

        assert out.out.shape[0] == demo_data["data"].shape[0]
        assert out.log_abs_det_jac.shape[0] == demo_data["data"].shape[0]
        assert out.out.shape[1:] == np.prod(made.model_config.output_dim)
        assert out.log_abs_det_jac.shape[1:] == np.prod(made.model_config.output_dim)



@pytest.mark.slow
class Test_MADE_Training:
    @pytest.fixture
    def train_dataset(self):
        return torch.load(os.path.join(PATH, "data/mnist_clean_train_dataset_sample"))

    @pytest.fixture(
        params=[BaseTrainerConfig(num_epochs=3, steps_saving=2, learning_rate=1e-5)]
    )
    def training_configs(self, tmpdir, request):
        tmpdir.mkdir("dummy_folder")
        dir_path = os.path.join(tmpdir, "dummy_folder")
        request.param.output_dir = dir_path
        return request.param

    @pytest.fixture(
    params=[
        MADEConfig(input_dim=(784,), output_dim=(784,), degrees_ordering='random'),
    ]
    )
    def model_configs(self, request):
        return request.param

    @pytest.fixture
    def made(self, model_configs):
        model = MADE(model_configs)
        return model

    @pytest.fixture()
    def prior(self, model_configs, request):

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        return  torch.distributions.Normal(
            torch.zeros(np.prod(model_configs.input_dim)).to(device),
            torch.ones(np.prod(model_configs.input_dim)).to(device)
            )

    @pytest.fixture(params=[Adam])
    def optimizers(self, request, made, training_configs):
        if request.param is not None:
            optimizer = request.param(
                made.parameters(), lr=training_configs.learning_rate
            )

        else:
            optimizer = None

        return optimizer

    def test_made_train_step(self, made, prior, train_dataset, training_configs, optimizers):

        nf_model = NFModel(prior=prior, flow=made)

        trainer = BaseTrainer(
            model=nf_model,
            train_dataset=train_dataset,
            training_config=training_configs,
            optimizer=optimizers,
        )

        start_model_state_dict = deepcopy(trainer.model.state_dict())

        step_1_loss = trainer.train_step(epoch=1)

        step_1_model_state_dict = deepcopy(trainer.model.state_dict())

        # check that weights were updated
        assert not all(
            [
                torch.equal(start_model_state_dict[key], step_1_model_state_dict[key])
                for key in start_model_state_dict.keys()
            ]
        )

    def test_made_eval_step(self, made, prior, train_dataset, training_configs, optimizers):

        nf_model = NFModel(prior=prior, flow=made)

        trainer = BaseTrainer(
            model=nf_model,
            train_dataset=train_dataset,
            eval_dataset=train_dataset,
            training_config=training_configs,
            optimizer=optimizers,
        )

        start_model_state_dict = deepcopy(trainer.model.state_dict())

        step_1_loss = trainer.eval_step(epoch=1)

        step_1_model_state_dict = deepcopy(trainer.model.state_dict())

        # check that weights were updated
        assert all(
            [
                torch.equal(start_model_state_dict[key], step_1_model_state_dict[key])
                for key in start_model_state_dict.keys()
            ]
        )

    def test_made_main_train_loop(
        self, made, prior, train_dataset, training_configs, optimizers):

        nf_model = NFModel(prior=prior, flow=made)

        trainer = BaseTrainer(
            model=nf_model,
            train_dataset=train_dataset,
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

    def test_checkpoint_saving(
        self, tmpdir, made, prior, train_dataset, training_configs, optimizers
    ):

        dir_path = training_configs.output_dir

        nf_model = NFModel(prior=prior, flow=made)

        trainer = BaseTrainer(
            model=nf_model,
            train_dataset=train_dataset,
            training_config=training_configs,
            optimizer=optimizers,
        )

        # Make a training step
        step_1_loss = trainer.train_step(epoch=1)

        model = deepcopy(trainer.model.flow)
        optimizer = deepcopy(trainer.optimizer)

        trainer.save_checkpoint(dir_path=dir_path, epoch=0, model=model)

        checkpoint_dir = os.path.join(dir_path, "checkpoint_epoch_0")

        assert os.path.isdir(checkpoint_dir)

        files_list = os.listdir(checkpoint_dir)

        assert set(["model.pt", "optimizer.pt", "training_config.json"]).issubset(
            set(files_list)
        )


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
        model_rec = MADE.load_from_folder(os.path.join(checkpoint_dir))

        assert all(
            [
                torch.equal(
                    model_rec.state_dict()[key].cpu(), model.state_dict()[key].cpu()
                )
                for key in model.state_dict().keys()
            ]
        )

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
        self, tmpdir, made, prior, train_dataset, training_configs, optimizers
    ):
        #
        target_saving_epoch = training_configs.steps_saving

        dir_path = training_configs.output_dir

        nf_model = NFModel(prior=prior, flow=made)

        trainer = BaseTrainer(
            model=nf_model,
            train_dataset=train_dataset,
            training_config=training_configs,
            optimizer=optimizers,
        )

        model = deepcopy(trainer.model.flow)

        trainer.train()

        training_dir = os.path.join(
            dir_path, f"MADE_training_{trainer._training_signature}"
        )
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
        self, tmpdir, made, prior, train_dataset, training_configs, optimizers
    ):

        dir_path = training_configs.output_dir

        nf_model = NFModel(prior=prior, flow=made)

        trainer = BaseTrainer(
            model=nf_model,
            train_dataset=train_dataset,
            training_config=training_configs,
            optimizer=optimizers,
        )

        trainer.train()

        model = deepcopy(trainer._best_model.flow)

        training_dir = os.path.join(
            dir_path, f"MADE_training_{trainer._training_signature}"
        )
        assert os.path.isdir(training_dir)

        final_dir = os.path.join(training_dir, f"final_model")
        assert os.path.isdir(final_dir)

        files_list = os.listdir(final_dir)

        assert set(["model.pt", "model_config.json", "training_config.json"]).issubset(
            set(files_list)
        )


        # check reload full model
        model_rec = MADE.load_from_folder(os.path.join(final_dir))

        assert all(
            [
                torch.equal(
                    model_rec.state_dict()[key].cpu(), model.state_dict()[key].cpu()
                )
                for key in model.state_dict().keys()
            ]
        )

#    def test_made_training_pipeline(self, tmpdir, made, train_dataset, training_configs):
#
#        dir_path = training_configs.output_dir
#
#        # build pipeline
#        pipeline = TrainingPipeline(model=made, training_config=training_configs)
#
#        assert pipeline.training_config.__dict__ == training_configs.__dict__
#
#        # Launch Pipeline
#        pipeline(
#            train_data=train_dataset.data,  # gives tensor to pipeline
#            eval_data=train_dataset.data,  # gives tensor to pipeline
#        )
#
#        model = deepcopy(pipeline.trainer._best_model)
#
#        training_dir = os.path.join(
#            dir_path, f"MADE_training_{pipeline.trainer._training_signature}"
#        )
#        assert os.path.isdir(training_dir)
#
#        final_dir = os.path.join(training_dir, f"final_model")
#        assert os.path.isdir(final_dir)
#
#        files_list = os.listdir(final_dir)
#
#        assert set(["model.pt", "model_config.json", "training_config.json"]).issubset(
#            set(files_list)
#        )
#
#        # check pickled custom decoder
#        if not made.model_config.uses_default_decoder:
#            assert "decoder.pkl" in files_list
#
#        else:
#            assert not "decoder.pkl" in files_list
#
#        # check pickled custom encoder
#        if not made.model_config.uses_default_encoder:
#            assert "encoder.pkl" in files_list
#
#        else:
#            assert not "encoder.pkl" in files_list
#
#        # check reload full model
#        model_rec = MADE.load_from_folder(os.path.join(final_dir))
#
#        assert all(
#            [
#                torch.equal(
#                    model_rec.state_dict()[key].cpu(), model.state_dict()[key].cpu()
#                )
#                for key in model.state_dict().keys()
#            ]
#        )
#
#        assert type(model_rec.encoder.cpu()) == type(model.encoder.cpu())
#        assert type(model_rec.decoder.cpu()) == type(model.decoder.cpu())
#