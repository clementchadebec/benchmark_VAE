import os
from copy import deepcopy

import numpy as np
import pytest
import torch

from pythae.customexception import BadInheritanceError
from pythae.models import Adversarial_AE, Adversarial_AE_Config, AutoModel
from pythae.models.base.base_utils import ModelOutput
from pythae.pipelines import GenerationPipeline, TrainingPipeline
from pythae.samplers import (
    GaussianMixtureSamplerConfig,
    IAFSamplerConfig,
    MAFSamplerConfig,
    NormalSamplerConfig,
    TwoStageVAESamplerConfig,
)
from pythae.trainers import (
    AdversarialTrainer,
    AdversarialTrainerConfig,
    BaseTrainerConfig,
)
from tests.data.custom_architectures import (
    Decoder_AE_Conv,
    Discriminator_MLP_Custom,
    Encoder_VAE_Conv,
    NetBadInheritance,
)

PATH = os.path.dirname(os.path.abspath(__file__))

device = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(params=[Adversarial_AE_Config(), Adversarial_AE_Config(latent_dim=5)])
def model_configs_no_input_dim(request):
    return request.param


@pytest.fixture(
    params=[
        Adversarial_AE_Config(
            input_dim=(1, 28, 28), latent_dim=10, reconstruction_loss="bce"
        ),
        Adversarial_AE_Config(input_dim=(1, 2, 18), latent_dim=5),
    ]
)
def model_configs(request):
    return request.param


@pytest.fixture
def custom_encoder(model_configs):
    return Encoder_VAE_Conv(model_configs)


@pytest.fixture
def custom_decoder(model_configs):
    return Decoder_AE_Conv(model_configs)


@pytest.fixture
def custom_discriminator(model_configs):
    return Discriminator_MLP_Custom(model_configs)


class Test_Model_Building:
    @pytest.fixture()
    def bad_net(self):
        return NetBadInheritance()

    def test_build_model(self, model_configs):
        model = Adversarial_AE(model_configs)
        assert all(
            [
                model.input_dim == model_configs.input_dim,
                model.latent_dim == model_configs.latent_dim,
            ]
        )

        # check raises error with adv scale > 1
        conf = Adversarial_AE_Config(
            input_dim=(1, 2, 18),
            latent_dim=5,
            adversarial_loss_scale=2 + np.random.rand(),
        )
        with pytest.raises(AssertionError):
            Adversarial_AE(conf)

        # check raises error with adv scale < 0
        conf = Adversarial_AE_Config(
            input_dim=(1, 2, 18),
            latent_dim=5,
            adversarial_loss_scale=2 + np.random.rand(),
        )
        with pytest.raises(AssertionError):
            a = Adversarial_AE(conf)

    def test_raises_bad_inheritance(self, model_configs, bad_net):
        with pytest.raises(BadInheritanceError):
            adversarial_ae = Adversarial_AE(model_configs, encoder=bad_net)

        with pytest.raises(BadInheritanceError):
            adversarial_ae = Adversarial_AE(model_configs, decoder=bad_net)

        with pytest.raises(BadInheritanceError):
            adversarial_ae = Adversarial_AE(model_configs, discriminator=bad_net)

    def test_raises_no_input_dim(
        self,
        model_configs_no_input_dim,
        custom_encoder,
        custom_decoder,
        custom_discriminator,
    ):
        with pytest.raises(AttributeError):
            adversarial_ae = Adversarial_AE(model_configs_no_input_dim)

        with pytest.raises(AttributeError):
            adversarial_ae = Adversarial_AE(
                model_configs_no_input_dim, encoder=custom_encoder
            )

        with pytest.raises(AttributeError):
            adversarial_ae = Adversarial_AE(
                model_configs_no_input_dim, decoder=custom_decoder
            )

        with pytest.raises(AttributeError):
            adversarial_ae = Adversarial_AE(
                model_configs_no_input_dim, discriminator=custom_discriminator
            )

        adversarial_ae = Adversarial_AE(
            model_configs_no_input_dim,
            encoder=custom_encoder,
            decoder=custom_decoder,
            discriminator=custom_discriminator,
        )

    def test_build_custom_arch(
        self, model_configs, custom_encoder, custom_decoder, custom_discriminator
    ):

        adversarial_ae = Adversarial_AE(
            model_configs, encoder=custom_encoder, decoder=custom_decoder
        )

        assert adversarial_ae.encoder == custom_encoder
        assert not adversarial_ae.model_config.uses_default_encoder

        assert adversarial_ae.decoder == custom_decoder
        assert not adversarial_ae.model_config.uses_default_encoder

        assert adversarial_ae.model_config.uses_default_discriminator

        adversarial_ae = Adversarial_AE(
            model_configs, discriminator=custom_discriminator
        )

        assert adversarial_ae.model_config.uses_default_encoder
        assert adversarial_ae.model_config.uses_default_encoder

        assert adversarial_ae.discriminator == custom_discriminator
        assert not adversarial_ae.model_config.uses_default_discriminator


class Test_Model_Saving:
    def test_default_model_saving(self, tmpdir, model_configs):

        tmpdir.mkdir("dummy_folder")
        dir_path = dir_path = os.path.join(tmpdir, "dummy_folder")

        model = Adversarial_AE(model_configs)

        model.state_dict()["encoder.layers.0.0.weight"][0] = 0

        model.save(dir_path=dir_path)

        assert set(os.listdir(dir_path)) == set(
            ["model_config.json", "model.pt", "environment.json"]
        )

        # reload model
        model_rec = AutoModel.load_from_folder(dir_path)

        # check configs are the same
        assert model_rec.model_config.__dict__ == model.model_config.__dict__

        assert all(
            [
                torch.equal(model_rec.state_dict()[key], model.state_dict()[key])
                for key in model.state_dict().keys()
            ]
        )

    def test_custom_encoder_model_saving(self, tmpdir, model_configs, custom_encoder):

        tmpdir.mkdir("dummy_folder")
        dir_path = dir_path = os.path.join(tmpdir, "dummy_folder")

        model = Adversarial_AE(model_configs, encoder=custom_encoder)

        model.state_dict()["encoder.layers.0.0.weight"][0] = 0

        model.save(dir_path=dir_path)

        assert set(os.listdir(dir_path)) == set(
            ["model_config.json", "model.pt", "encoder.pkl", "environment.json"]
        )

        # reload model
        model_rec = AutoModel.load_from_folder(dir_path)

        # check configs are the same
        assert model_rec.model_config.__dict__ == model.model_config.__dict__

        assert all(
            [
                torch.equal(model_rec.state_dict()[key], model.state_dict()[key])
                for key in model.state_dict().keys()
            ]
        )

    def test_custom_decoder_model_saving(self, tmpdir, model_configs, custom_decoder):

        tmpdir.mkdir("dummy_folder")
        dir_path = dir_path = os.path.join(tmpdir, "dummy_folder")

        model = Adversarial_AE(model_configs, decoder=custom_decoder)

        model.state_dict()["encoder.layers.0.0.weight"][0] = 0

        model.save(dir_path=dir_path)

        assert set(os.listdir(dir_path)) == set(
            ["model_config.json", "model.pt", "decoder.pkl", "environment.json"]
        )

        # reload model
        model_rec = AutoModel.load_from_folder(dir_path)

        # check configs are the same
        assert model_rec.model_config.__dict__ == model.model_config.__dict__

        assert all(
            [
                torch.equal(model_rec.state_dict()[key], model.state_dict()[key])
                for key in model.state_dict().keys()
            ]
        )

    def test_custom_discriminator_model_saving(
        self, tmpdir, model_configs, custom_discriminator
    ):

        tmpdir.mkdir("dummy_folder")
        dir_path = dir_path = os.path.join(tmpdir, "dummy_folder")

        model = Adversarial_AE(model_configs, discriminator=custom_discriminator)

        model.state_dict()["encoder.layers.0.0.weight"][0] = 0

        model.save(dir_path=dir_path)

        assert set(os.listdir(dir_path)) == set(
            ["model_config.json", "model.pt", "discriminator.pkl", "environment.json"]
        )

        # reload model
        model_rec = AutoModel.load_from_folder(dir_path)

        # check configs are the same
        assert model_rec.model_config.__dict__ == model.model_config.__dict__

        assert all(
            [
                torch.equal(model_rec.state_dict()[key], model.state_dict()[key])
                for key in model.state_dict().keys()
            ]
        )

    def test_full_custom_model_saving(
        self,
        tmpdir,
        model_configs,
        custom_encoder,
        custom_decoder,
        custom_discriminator,
    ):

        tmpdir.mkdir("dummy_folder")
        dir_path = dir_path = os.path.join(tmpdir, "dummy_folder")

        model = Adversarial_AE(
            model_configs,
            encoder=custom_encoder,
            decoder=custom_decoder,
            discriminator=custom_discriminator,
        )

        model.state_dict()["encoder.layers.0.0.weight"][0] = 0

        model.save(dir_path=dir_path)

        assert set(os.listdir(dir_path)) == set(
            [
                "model_config.json",
                "model.pt",
                "encoder.pkl",
                "decoder.pkl",
                "discriminator.pkl",
                "environment.json",
            ]
        )

        # reload model
        model_rec = AutoModel.load_from_folder(dir_path)

        # check configs are the same
        assert model_rec.model_config.__dict__ == model.model_config.__dict__

        assert all(
            [
                torch.equal(model_rec.state_dict()[key], model.state_dict()[key])
                for key in model.state_dict().keys()
            ]
        )

    def test_raises_missing_files(
        self,
        tmpdir,
        model_configs,
        custom_encoder,
        custom_decoder,
        custom_discriminator,
    ):

        tmpdir.mkdir("dummy_folder")
        dir_path = dir_path = os.path.join(tmpdir, "dummy_folder")

        model = Adversarial_AE(
            model_configs,
            encoder=custom_encoder,
            decoder=custom_decoder,
            discriminator=custom_discriminator,
        )

        model.state_dict()["encoder.layers.0.0.weight"][0] = 0

        model.save(dir_path=dir_path)

        os.remove(os.path.join(dir_path, "discriminator.pkl"))

        # check raises decoder.pkl is missing
        with pytest.raises(FileNotFoundError):
            model_rec = AutoModel.load_from_folder(dir_path)

        os.remove(os.path.join(dir_path, "decoder.pkl"))

        # check raises decoder.pkl is missing
        with pytest.raises(FileNotFoundError):
            model_rec = AutoModel.load_from_folder(dir_path)

        os.remove(os.path.join(dir_path, "encoder.pkl"))

        # check raises encoder.pkl is missing
        with pytest.raises(FileNotFoundError):
            model_rec = AutoModel.load_from_folder(dir_path)

        os.remove(os.path.join(dir_path, "model.pt"))

        # check raises encoder.pkl is missing
        with pytest.raises(FileNotFoundError):
            model_rec = AutoModel.load_from_folder(dir_path)

        os.remove(os.path.join(dir_path, "model_config.json"))

        # check raises encoder.pkl is missing
        with pytest.raises(FileNotFoundError):
            model_rec = AutoModel.load_from_folder(dir_path)


class Test_Model_forward:
    @pytest.fixture
    def demo_data(self):
        data = torch.load(os.path.join(PATH, "data/mnist_clean_train_dataset_sample"))[
            :
        ]
        return data  # This is an extract of 3 data from MNIST (unnormalized) used to test custom architecture

    @pytest.fixture
    def adversarial_ae(self, model_configs, demo_data):
        model_configs.input_dim = tuple(demo_data["data"][0].shape)
        return Adversarial_AE(model_configs)

    def test_model_train_output(self, adversarial_ae, demo_data):

        # model_configs.input_dim = demo_data['data'][0].shape[-1]

        # adversarial_ae = Adversarial_AE(model_configs)

        adversarial_ae.train()

        out = adversarial_ae(demo_data)

        assert isinstance(out, ModelOutput)

        assert (
            set(
                [
                    "loss",
                    "recon_loss",
                    "autoencoder_loss",
                    "discriminator_loss",
                    "recon_x",
                    "z",
                ]
            )
            == set(out.keys())
        )

        assert out.z.shape[0] == demo_data["data"].shape[0]
        assert out.recon_x.shape == demo_data["data"].shape


class Test_Model_interpolate:
    @pytest.fixture(
        params=[
            torch.randn(3, 2, 3, 1),
            torch.randn(3, 2, 2),
            torch.load(os.path.join(PATH, "data/mnist_clean_train_dataset_sample"))[:][
                "data"
            ],
        ]
    )
    def demo_data(self, request):
        return request.param

    @pytest.fixture()
    def granularity(self):
        return int(torch.randint(1, 10, (1,)))

    @pytest.fixture
    def adversarial_ae(self, model_configs, demo_data):
        model_configs.input_dim = tuple(demo_data[0].shape)
        return Adversarial_AE(model_configs)

    def test_interpolate(self, adversarial_ae, demo_data, granularity):
        with pytest.raises(AssertionError):
            adversarial_ae.interpolate(demo_data, demo_data[1:], granularity)

        interp = adversarial_ae.interpolate(demo_data, demo_data, granularity)

        assert (
            tuple(interp.shape)
            == (
                demo_data.shape[0],
                granularity,
            )
            + (demo_data.shape[1:])
        )


class Test_Model_reconstruct:
    @pytest.fixture(
        params=[
            torch.randn(3, 2, 3, 1),
            torch.randn(3, 2, 2),
            torch.load(os.path.join(PATH, "data/mnist_clean_train_dataset_sample"))[:][
                "data"
            ],
        ]
    )
    def demo_data(self, request):
        return request.param

    @pytest.fixture
    def adversarial_ae(self, model_configs, demo_data):
        model_configs.input_dim = tuple(demo_data[0].shape)
        return Adversarial_AE(model_configs)

    def test_reconstruct(self, adversarial_ae, demo_data):

        recon = adversarial_ae.reconstruct(demo_data)
        assert tuple(recon.shape) == demo_data.shape


@pytest.mark.slow
class Test_Adversarial_AE_Training:
    @pytest.fixture
    def train_dataset(self):
        return torch.load(os.path.join(PATH, "data/mnist_clean_train_dataset_sample"))

    @pytest.fixture(
        params=[
            AdversarialTrainerConfig(num_epochs=3, steps_saving=2, learning_rate=1e-5)
        ]
    )
    def training_configs(self, tmpdir, request):
        tmpdir.mkdir("dummy_folder")
        dir_path = os.path.join(tmpdir, "dummy_folder")
        request.param.output_dir = dir_path
        return request.param

    @pytest.fixture(
        params=[
            torch.rand(1),
            torch.rand(1),
            torch.rand(1),
            torch.rand(1),
            torch.rand(1),
        ]
    )
    def adversarial_ae(
        self,
        model_configs,
        custom_encoder,
        custom_decoder,
        custom_discriminator,
        request,
    ):
        # randomized

        alpha = request.param

        if alpha < 0.125:
            model = Adversarial_AE(model_configs)

        elif 0.125 <= alpha < 0.25:
            model = Adversarial_AE(model_configs, encoder=custom_encoder)

        elif 0.25 <= alpha < 0.375:
            model = Adversarial_AE(model_configs, decoder=custom_decoder)

        elif 0.375 <= alpha < 0.5:
            model = Adversarial_AE(model_configs, discriminator=custom_discriminator)

        elif 0.5 <= alpha < 0.625:
            model = Adversarial_AE(
                model_configs, encoder=custom_encoder, decoder=custom_decoder
            )

        elif 0.625 <= alpha < 0:
            model = Adversarial_AE(
                model_configs,
                encoder=custom_encoder,
                discriminator=custom_discriminator,
            )

        elif 0.750 <= alpha < 0.875:
            model = Adversarial_AE(
                model_configs,
                decoder=custom_decoder,
                discriminator=custom_discriminator,
            )

        else:
            model = Adversarial_AE(
                model_configs,
                encoder=custom_encoder,
                decoder=custom_decoder,
                discriminator=custom_discriminator,
            )

        return model

    @pytest.fixture
    def trainer(self, adversarial_ae, train_dataset, training_configs):
        trainer = AdversarialTrainer(
            model=adversarial_ae,
            train_dataset=train_dataset,
            eval_dataset=train_dataset,
            training_config=training_configs,
        )

        trainer.prepare_training()

        return trainer

    def test_adversarial_ae_train_step(self, trainer):

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

    def test_adversarial_ae_eval_step(self, trainer):

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

    def test_adversarial_ae_predict_step(self, trainer, train_dataset):

        start_model_state_dict = deepcopy(trainer.model.state_dict())

        inputs, recon, generated = trainer.predict(trainer.model)

        step_1_model_state_dict = deepcopy(trainer.model.state_dict())

        # check that weights were not updated
        assert all(
            [
                torch.equal(start_model_state_dict[key], step_1_model_state_dict[key])
                for key in start_model_state_dict.keys()
            ]
        )

        assert inputs.cpu() in train_dataset.data
        assert recon.shape == inputs.shape
        assert generated.shape == inputs.shape

    def test_adversarial_ae_main_train_loop(self, trainer):

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

    def test_checkpoint_saving(self, adversarial_ae, trainer, training_configs):

        dir_path = training_configs.output_dir

        # Make a training step
        step_1_loss = trainer.train_step(epoch=1)

        model = deepcopy(trainer.model)
        autoencoder_optimizer = deepcopy(trainer.autoencoder_optimizer)
        discriminator_optimizer = deepcopy(trainer.discriminator_optimizer)

        trainer.save_checkpoint(dir_path=dir_path, epoch=0, model=model)

        checkpoint_dir = os.path.join(dir_path, "checkpoint_epoch_0")

        assert os.path.isdir(checkpoint_dir)

        files_list = os.listdir(checkpoint_dir)

        assert set(
            [
                "model.pt",
                "autoencoder_optimizer.pt",
                "discriminator_optimizer.pt",
                "training_config.json",
            ]
        ).issubset(set(files_list))

        # check pickled custom decoder
        if not adversarial_ae.model_config.uses_default_decoder:
            assert "decoder.pkl" in files_list

        else:
            assert not "decoder.pkl" in files_list

        # check pickled custom encoder
        if not adversarial_ae.model_config.uses_default_encoder:
            assert "encoder.pkl" in files_list

        else:
            assert not "encoder.pkl" in files_list

        # check pickled custom discriminator
        if not adversarial_ae.model_config.uses_default_discriminator:
            assert "discriminator.pkl" in files_list

        else:
            assert not "discriminator.pkl" in files_list

        model_rec_state_dict = torch.load(os.path.join(checkpoint_dir, "model.pt"))[
            "model_state_dict"
        ]

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
        model_rec = AutoModel.load_from_folder(os.path.join(checkpoint_dir))

        assert all(
            [
                torch.equal(
                    model_rec.state_dict()[key].cpu(), model.state_dict()[key].cpu()
                )
                for key in model.state_dict().keys()
            ]
        )

        assert type(model_rec.encoder.cpu()) == type(model.encoder.cpu())
        assert type(model_rec.decoder.cpu()) == type(model.decoder.cpu())
        assert type(model_rec.discriminator.cpu()) == type(model.discriminator.cpu())

        autoencoder_optim_rec_state_dict = torch.load(
            os.path.join(checkpoint_dir, "autoencoder_optimizer.pt")
        )
        discriminator_optim_rec_state_dict = torch.load(
            os.path.join(checkpoint_dir, "discriminator_optimizer.pt")
        )

        assert all(
            [
                dict_rec == dict_optimizer
                for (dict_rec, dict_optimizer) in zip(
                    autoencoder_optim_rec_state_dict["param_groups"],
                    autoencoder_optimizer.state_dict()["param_groups"],
                )
            ]
        )

        assert all(
            [
                dict_rec == dict_optimizer
                for (dict_rec, dict_optimizer) in zip(
                    autoencoder_optim_rec_state_dict["state"],
                    autoencoder_optimizer.state_dict()["state"],
                )
            ]
        )

        assert all(
            [
                dict_rec == dict_optimizer
                for (dict_rec, dict_optimizer) in zip(
                    discriminator_optim_rec_state_dict["param_groups"],
                    discriminator_optimizer.state_dict()["param_groups"],
                )
            ]
        )

        assert all(
            [
                dict_rec == dict_optimizer
                for (dict_rec, dict_optimizer) in zip(
                    discriminator_optim_rec_state_dict["state"],
                    discriminator_optimizer.state_dict()["state"],
                )
            ]
        )

    def test_checkpoint_saving_during_training(
        self, adversarial_ae, trainer, training_configs
    ):
        #
        target_saving_epoch = training_configs.steps_saving

        dir_path = training_configs.output_dir

        model = deepcopy(trainer.model)

        trainer.train()

        training_dir = os.path.join(
            dir_path, f"Adversarial_AE_training_{trainer._training_signature}"
        )
        assert os.path.isdir(training_dir)

        checkpoint_dir = os.path.join(
            training_dir, f"checkpoint_epoch_{target_saving_epoch}"
        )

        assert os.path.isdir(checkpoint_dir)

        files_list = os.listdir(checkpoint_dir)

        # check files
        assert set(
            [
                "model.pt",
                "autoencoder_optimizer.pt",
                "discriminator_optimizer.pt",
                "training_config.json",
            ]
        ).issubset(set(files_list))

        # check pickled custom decoder
        if not adversarial_ae.model_config.uses_default_decoder:
            assert "decoder.pkl" in files_list

        else:
            assert not "decoder.pkl" in files_list

        # check pickled custom encoder
        if not adversarial_ae.model_config.uses_default_encoder:
            assert "encoder.pkl" in files_list

        else:
            assert not "encoder.pkl" in files_list

        # check pickled custom discriminator
        if not adversarial_ae.model_config.uses_default_discriminator:
            assert "discriminator.pkl" in files_list

        else:
            assert not "discriminator.pkl" in files_list

        model_rec_state_dict = torch.load(os.path.join(checkpoint_dir, "model.pt"))[
            "model_state_dict"
        ]

        assert not all(
            [
                torch.equal(model_rec_state_dict[key], model.state_dict()[key])
                for key in model.state_dict().keys()
            ]
        )

    def test_final_model_saving(self, adversarial_ae, trainer, training_configs):

        dir_path = training_configs.output_dir

        trainer.train()

        model = deepcopy(trainer._best_model)

        training_dir = os.path.join(
            dir_path, f"Adversarial_AE_training_{trainer._training_signature}"
        )
        assert os.path.isdir(training_dir)

        final_dir = os.path.join(training_dir, f"final_model")
        assert os.path.isdir(final_dir)

        files_list = os.listdir(final_dir)

        assert set(["model.pt", "model_config.json", "training_config.json"]).issubset(
            set(files_list)
        )

        # check pickled custom decoder
        if not adversarial_ae.model_config.uses_default_decoder:
            assert "decoder.pkl" in files_list

        else:
            assert not "decoder.pkl" in files_list

        # check pickled custom encoder
        if not adversarial_ae.model_config.uses_default_encoder:
            assert "encoder.pkl" in files_list

        else:
            assert not "encoder.pkl" in files_list

        # check pickled custom discriminator
        if not adversarial_ae.model_config.uses_default_discriminator:
            assert "discriminator.pkl" in files_list

        else:
            assert not "discriminator.pkl" in files_list

        # check reload full model
        model_rec = AutoModel.load_from_folder(os.path.join(final_dir))

        assert all(
            [
                torch.equal(
                    model_rec.state_dict()[key].cpu(), model.state_dict()[key].cpu()
                )
                for key in model.state_dict().keys()
            ]
        )

        assert type(model_rec.encoder.cpu()) == type(model.encoder.cpu())
        assert type(model_rec.decoder.cpu()) == type(model.decoder.cpu())
        assert type(model_rec.discriminator.cpu()) == type(model.discriminator.cpu())

    def test_adversarial_ae_training_pipeline(
        self, adversarial_ae, train_dataset, training_configs
    ):

        with pytest.raises(AssertionError):
            pipeline = TrainingPipeline(
                model=adversarial_ae, training_config=BaseTrainerConfig()
            )

        dir_path = training_configs.output_dir

        # build pipeline
        pipeline = TrainingPipeline(
            model=adversarial_ae, training_config=training_configs
        )

        # Launch Pipeline
        pipeline(
            train_data=train_dataset.data,  # gives tensor to pipeline
            eval_data=train_dataset.data,  # gives tensor to pipeline
        )

        model = deepcopy(pipeline.trainer._best_model)

        training_dir = os.path.join(
            dir_path, f"Adversarial_AE_training_{pipeline.trainer._training_signature}"
        )
        assert os.path.isdir(training_dir)

        final_dir = os.path.join(training_dir, f"final_model")
        assert os.path.isdir(final_dir)

        files_list = os.listdir(final_dir)

        assert set(["model.pt", "model_config.json", "training_config.json"]).issubset(
            set(files_list)
        )

        # check pickled custom decoder
        if not adversarial_ae.model_config.uses_default_decoder:
            assert "decoder.pkl" in files_list

        else:
            assert not "decoder.pkl" in files_list

        # check pickled custom encoder
        if not adversarial_ae.model_config.uses_default_encoder:
            assert "encoder.pkl" in files_list

        else:
            assert not "encoder.pkl" in files_list

        # check pickled custom discriminator
        if not adversarial_ae.model_config.uses_default_discriminator:
            assert "discriminator.pkl" in files_list

        else:
            assert not "discriminator.pkl" in files_list

        # check reload full model
        model_rec = AutoModel.load_from_folder(os.path.join(final_dir))

        assert all(
            [
                torch.equal(
                    model_rec.state_dict()[key].cpu(), model.state_dict()[key].cpu()
                )
                for key in model.state_dict().keys()
            ]
        )

        assert type(model_rec.encoder.cpu()) == type(model.encoder.cpu())
        assert type(model_rec.decoder.cpu()) == type(model.decoder.cpu())
        assert type(model_rec.discriminator.cpu()) == type(model.discriminator.cpu())


class Test_Adversarial_AE_Generation:
    @pytest.fixture
    def train_data(self):
        return torch.load(
            os.path.join(PATH, "data/mnist_clean_train_dataset_sample")
        ).data

    @pytest.fixture()
    def ae_model(self):
        return Adversarial_AE(
            Adversarial_AE_Config(input_dim=(1, 28, 28), latent_dim=7)
        )

    @pytest.fixture(
        params=[
            NormalSamplerConfig(),
            GaussianMixtureSamplerConfig(),
            MAFSamplerConfig(),
            IAFSamplerConfig(),
            TwoStageVAESamplerConfig(),
        ]
    )
    def sampler_configs(self, request):
        return request.param

    def test_fits_in_generation_pipeline(self, ae_model, sampler_configs, train_data):
        pipeline = GenerationPipeline(model=ae_model, sampler_config=sampler_configs)
        gen_data = pipeline(
            num_samples=11,
            batch_size=7,
            output_dir=None,
            return_gen=True,
            train_data=train_data,
            eval_data=train_data,
            training_config=BaseTrainerConfig(num_epochs=1),
        )

        assert gen_data.shape[0] == 11
