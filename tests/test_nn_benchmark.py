import numpy as np
import pytest
import torch

from pythae.models import AEConfig, VAEConfig
from pythae.models.nn.benchmarks.celeba import *
from pythae.models.nn.benchmarks.cifar import *
from pythae.models.nn.benchmarks.mnist import *
from pythae.models.nn.default_architectures import *

device = "cuda" if torch.cuda.is_available() else "cpu"

#### MNIST configs ####
@pytest.fixture(
    params=[
        AEConfig(input_dim=(1, 28, 28), latent_dim=10),
        AEConfig(input_dim=(1, 28, 28), latent_dim=5),
    ]
)
def ae_mnist_config(request):
    return request.param


@pytest.fixture()
def mnist_like_data():
    return torch.rand(3, 1, 28, 28).to(device)


#### CIFAR configs ####
@pytest.fixture(
    params=[
        AEConfig(input_dim=(3, 32, 32), latent_dim=10),
        AEConfig(input_dim=(3, 32, 32), latent_dim=5),
    ]
)
def ae_cifar_config(request):
    return request.param


@pytest.fixture()
def cifar_like_data():
    return torch.rand(3, 3, 32, 32).to(device)


#### CELEBA configs ####
@pytest.fixture(
    params=[
        AEConfig(input_dim=(3, 64, 64), latent_dim=10),
        AEConfig(input_dim=(3, 64, 64), latent_dim=5),
    ]
)
def ae_celeba_config(request):
    return request.param


@pytest.fixture()
def celeba_like_data():
    return torch.rand(3, 3, 64, 64).to(device)


class Test_MNIST_Default:
    @pytest.fixture(params=[[1], None, [-1]])
    def recon_layers_default(self, request):
        return request.param

    def test_ae_encoding_decoding_default(
        self, ae_mnist_config, mnist_like_data, recon_layers_default
    ):
        encoder = Encoder_AE_MLP(ae_mnist_config).to(device)
        decoder = Decoder_AE_MLP(ae_mnist_config).to(device)

        embedding = encoder(mnist_like_data).embedding

        assert embedding.shape == (mnist_like_data.shape[0], ae_mnist_config.latent_dim)

        reconstruction = decoder(embedding).reconstruction

        assert reconstruction.shape == mnist_like_data.shape

        encoder_embed = encoder(
            mnist_like_data, output_layer_levels=recon_layers_default
        )
        decoder_recon = decoder(embedding, output_layer_levels=recon_layers_default)

        if recon_layers_default is None:
            assert "embedding" in encoder_embed.keys()
            assert "reconstruction" in decoder_recon.keys()

        else:
            for lev in recon_layers_default:
                if lev != -1:
                    assert f"embedding_layer_{lev}" in encoder_embed.keys()
                    assert f"reconstruction_layer_{lev}" in decoder_recon.keys()

                else:
                    assert "embedding" in encoder_embed.keys()
                    assert "reconstruction" in decoder_recon.keys()

        if recon_layers_default is None:
            assert encoder_embed["embedding"].shape[1] == ae_mnist_config.latent_dim
            assert (
                decoder_recon[f"reconstruction"].shape[1:] == ae_mnist_config.input_dim
            )

        else:
            if 1 in recon_layers_default:
                assert encoder_embed[f"embedding_layer_1"].shape[1] == 512
                assert decoder_recon[f"reconstruction_layer_1"].shape[1] == 512

            if -1 in recon_layers_default:
                assert encoder_embed["embedding"].shape[1] == ae_mnist_config.latent_dim
                assert (
                    decoder_recon[f"reconstruction"].shape[1:]
                    == ae_mnist_config.input_dim
                )

    def test_vae_encoding_decoding_default(
        self, ae_mnist_config, mnist_like_data, recon_layers_default
    ):
        encoder = Encoder_VAE_MLP(ae_mnist_config).to(device)
        decoder = Decoder_AE_MLP(ae_mnist_config).to(device)

        output = encoder(mnist_like_data)
        embedding, log_covariance = output.embedding, output.log_covariance

        assert embedding.shape == (
            mnist_like_data.shape[0],
            ae_mnist_config.latent_dim,
        )
        assert log_covariance.shape == (
            mnist_like_data.shape[0],
            ae_mnist_config.latent_dim,
        )

        reconstruction = decoder(embedding).reconstruction

        assert reconstruction.shape == mnist_like_data.shape

        encoder_embed = encoder(
            mnist_like_data, output_layer_levels=recon_layers_default
        )
        decoder_recon = decoder(embedding, output_layer_levels=recon_layers_default)

        if recon_layers_default is None:
            assert "embedding" in encoder_embed.keys()
            assert "log_covariance" in encoder_embed.keys()
            assert "reconstruction" in decoder_recon.keys()

        else:
            for lev in recon_layers_default:
                if lev != -1:
                    assert f"embedding_layer_{lev}" in encoder_embed.keys()
                    assert f"reconstruction_layer_{lev}" in decoder_recon.keys()

                else:
                    assert "embedding" in encoder_embed.keys()
                    assert "log_covariance" in encoder_embed.keys()
                    assert "reconstruction" in decoder_recon.keys()

        if recon_layers_default is None:
            assert encoder_embed["embedding"].shape[1] == ae_mnist_config.latent_dim
            assert (
                encoder_embed["log_covariance"].shape[1] == ae_mnist_config.latent_dim
            )
            assert (
                decoder_recon[f"reconstruction"].shape[1:] == ae_mnist_config.input_dim
            )

        else:
            if 1 in recon_layers_default:
                assert encoder_embed[f"embedding_layer_1"].shape[1] == 512
                assert decoder_recon[f"reconstruction_layer_1"].shape[1] == 512

            if -1 in recon_layers_default:
                assert encoder_embed["embedding"].shape[1] == ae_mnist_config.latent_dim
                assert (
                    encoder_embed["log_covariance"].shape[1]
                    == ae_mnist_config.latent_dim
                )
                assert (
                    decoder_recon[f"reconstruction"].shape[1:]
                    == ae_mnist_config.input_dim
                )

    def test_svae_encoding_decoding_default(
        self, ae_mnist_config, mnist_like_data, recon_layers_default
    ):
        encoder = Encoder_SVAE_MLP(ae_mnist_config).to(device)
        decoder = Decoder_AE_MLP(ae_mnist_config).to(device)

        output = encoder(mnist_like_data)
        embedding, log_concentration = output.embedding, output.log_concentration

        assert embedding.shape == (
            mnist_like_data.shape[0],
            ae_mnist_config.latent_dim,
        )
        assert log_concentration.shape == (mnist_like_data.shape[0], 1)

        reconstruction = decoder(embedding).reconstruction

        assert reconstruction.shape == mnist_like_data.shape

        encoder_embed = encoder(
            mnist_like_data, output_layer_levels=recon_layers_default
        )
        decoder_recon = decoder(embedding, output_layer_levels=recon_layers_default)

        if recon_layers_default is None:
            assert "embedding" in encoder_embed.keys()
            assert "log_concentration" in encoder_embed.keys()
            assert "reconstruction" in decoder_recon.keys()

        else:
            for lev in recon_layers_default:
                if lev != -1:
                    assert f"embedding_layer_{lev}" in encoder_embed.keys()
                    assert f"reconstruction_layer_{lev}" in decoder_recon.keys()

                else:
                    assert "embedding" in encoder_embed.keys()
                    assert "log_concentration" in encoder_embed.keys()
                    assert "reconstruction" in decoder_recon.keys()

        if recon_layers_default is None:
            assert encoder_embed["embedding"].shape[1] == ae_mnist_config.latent_dim
            assert encoder_embed["log_concentration"].shape[1] == 1
            assert (
                decoder_recon[f"reconstruction"].shape[1:] == ae_mnist_config.input_dim
            )

        else:
            if 1 in recon_layers_default:
                assert encoder_embed[f"embedding_layer_1"].shape[1] == 512
                assert decoder_recon[f"reconstruction_layer_1"].shape[1] == 512

            if -1 in recon_layers_default:
                assert encoder_embed["embedding"].shape[1] == ae_mnist_config.latent_dim
                assert encoder_embed["log_concentration"].shape[1] == 1
                assert (
                    decoder_recon[f"reconstruction"].shape[1:]
                    == ae_mnist_config.input_dim
                )

    def test_discriminator_default(
        self, ae_mnist_config, mnist_like_data, recon_layers_default
    ):

        ae_mnist_config.discriminator_input_dim = (1, 28, 28)

        discriminator = Discriminator_MLP(ae_mnist_config).to(device)

        scores = discriminator(
            mnist_like_data, output_layer_levels=recon_layers_default
        )

        if recon_layers_default is None:
            assert "embedding" in scores.keys()

        else:
            for lev in recon_layers_default:
                if lev != -1:
                    assert f"embedding_layer_{lev}" in scores.keys()
                else:
                    assert "embedding" in scores.keys()

        if recon_layers_default is None:
            assert scores["embedding"].shape[1] == 1

        else:
            if 1 in recon_layers_default:
                assert scores[f"embedding_layer_1"].shape[1] == 256

            if 2 in recon_layers_default:
                assert scores[f"embedding_layer_2"].shape[1] == 1

            if -1 in recon_layers_default:
                assert scores["embedding"].shape[1] == 1


class Test_MNIST_ConvNets:
    @pytest.fixture(params=[[3, 4], [np.random.randint(1, 5)], [1, 2, 4, -1], None])
    def recon_layers(self, request):
        return request.param

    def test_ae_encoding_decoding(self, ae_mnist_config, mnist_like_data, recon_layers):
        encoder = Encoder_Conv_AE_MNIST(ae_mnist_config).to(device)
        decoder = Decoder_Conv_AE_MNIST(ae_mnist_config).to(device)

        embedding = encoder(mnist_like_data).embedding

        assert embedding.shape == (mnist_like_data.shape[0], ae_mnist_config.latent_dim)

        reconstruction = decoder(embedding).reconstruction

        assert reconstruction.shape == mnist_like_data.shape

        encoder_embed = encoder(mnist_like_data, output_layer_levels=recon_layers)
        decoder_recon = decoder(embedding, output_layer_levels=recon_layers)

        if recon_layers is None:
            assert "embedding" in encoder_embed.keys()
            assert "reconstruction" in decoder_recon.keys()

        else:
            for lev in recon_layers:
                if lev != -1:
                    assert f"embedding_layer_{lev}" in encoder_embed.keys()
                    assert f"reconstruction_layer_{lev}" in decoder_recon.keys()

                else:
                    assert "embedding" in encoder_embed.keys()
                    assert "reconstruction" in decoder_recon.keys()

        if recon_layers is None:
            assert encoder_embed["embedding"].shape[1] == ae_mnist_config.latent_dim
            assert (
                decoder_recon[f"reconstruction"].shape[1:] == ae_mnist_config.input_dim
            )

        else:
            if 1 in recon_layers:
                assert encoder_embed[f"embedding_layer_1"].shape[1] == 128
                assert decoder_recon[f"reconstruction_layer_1"].shape[1] == 1024

            if 2 in recon_layers:
                assert encoder_embed[f"embedding_layer_2"].shape[1] == 256
                assert decoder_recon[f"reconstruction_layer_2"].shape[1] == 512

            if 3 in recon_layers:
                assert encoder_embed[f"embedding_layer_3"].shape[1] == 512
                assert decoder_recon[f"reconstruction_layer_3"].shape[1] == 256

            if 4 in recon_layers:
                assert encoder_embed[f"embedding_layer_4"].shape[1] == 1024
                assert (
                    decoder_recon[f"reconstruction_layer_4"].shape[1:]
                    == ae_mnist_config.input_dim
                )

            if -1 in recon_layers:
                assert encoder_embed["embedding"].shape[1] == ae_mnist_config.latent_dim
                assert (
                    decoder_recon[f"reconstruction"].shape[1:]
                    == ae_mnist_config.input_dim
                )

    def test_vae_encoding_decoding(
        self, ae_mnist_config, mnist_like_data, recon_layers
    ):
        encoder = Encoder_Conv_VAE_MNIST(ae_mnist_config).to(device)
        decoder = Decoder_Conv_AE_MNIST(ae_mnist_config).to(device)

        output = encoder(mnist_like_data)
        embedding, log_covariance = output.embedding, output.log_covariance

        assert embedding.shape == (
            mnist_like_data.shape[0],
            ae_mnist_config.latent_dim,
        )
        assert log_covariance.shape == (
            mnist_like_data.shape[0],
            ae_mnist_config.latent_dim,
        )

        reconstruction = decoder(embedding).reconstruction

        assert reconstruction.shape == mnist_like_data.shape

        encoder_embed = encoder(mnist_like_data, output_layer_levels=recon_layers)
        decoder_recon = decoder(embedding, output_layer_levels=recon_layers)

        if recon_layers is None:
            assert "embedding" in encoder_embed.keys()
            assert "log_covariance" in encoder_embed.keys()
            assert "reconstruction" in decoder_recon.keys()

        else:
            for lev in recon_layers:
                if lev != -1:
                    assert f"embedding_layer_{lev}" in encoder_embed.keys()
                    assert f"reconstruction_layer_{lev}" in decoder_recon.keys()

                else:
                    assert "embedding" in encoder_embed.keys()
                    assert "log_covariance" in encoder_embed.keys()
                    assert "reconstruction" in decoder_recon.keys()

        if recon_layers is None:
            assert encoder_embed["embedding"].shape[1] == ae_mnist_config.latent_dim
            assert (
                encoder_embed["log_covariance"].shape[1] == ae_mnist_config.latent_dim
            )
            assert (
                decoder_recon[f"reconstruction"].shape[1:] == ae_mnist_config.input_dim
            )

        else:
            if 1 in recon_layers:
                assert encoder_embed[f"embedding_layer_1"].shape[1] == 128
                assert decoder_recon[f"reconstruction_layer_1"].shape[1] == 1024

            if 2 in recon_layers:
                assert encoder_embed[f"embedding_layer_2"].shape[1] == 256
                assert decoder_recon[f"reconstruction_layer_2"].shape[1] == 512

            if 3 in recon_layers:
                assert encoder_embed[f"embedding_layer_3"].shape[1] == 512
                assert decoder_recon[f"reconstruction_layer_3"].shape[1] == 256

            if 4 in recon_layers:
                assert encoder_embed[f"embedding_layer_4"].shape[1] == 1024
                assert (
                    decoder_recon[f"reconstruction_layer_4"].shape[1:]
                    == ae_mnist_config.input_dim
                )

            if -1 in recon_layers:
                assert encoder_embed["embedding"].shape[1] == ae_mnist_config.latent_dim
                assert (
                    encoder_embed["log_covariance"].shape[1]
                    == ae_mnist_config.latent_dim
                )
                assert (
                    decoder_recon[f"reconstruction"].shape[1:]
                    == ae_mnist_config.input_dim
                )

    def test_svae_encoding_decoding(
        self, ae_mnist_config, mnist_like_data, recon_layers
    ):
        encoder = Encoder_Conv_SVAE_MNIST(ae_mnist_config).to(device)
        decoder = Decoder_Conv_AE_MNIST(ae_mnist_config).to(device)

        output = encoder(mnist_like_data)
        embedding, log_concentration = output.embedding, output.log_concentration

        assert embedding.shape == (
            mnist_like_data.shape[0],
            ae_mnist_config.latent_dim,
        )
        assert log_concentration.shape == (mnist_like_data.shape[0], 1)

        reconstruction = decoder(embedding).reconstruction

        assert reconstruction.shape == mnist_like_data.shape

        encoder_embed = encoder(mnist_like_data, output_layer_levels=recon_layers)
        decoder_recon = decoder(embedding, output_layer_levels=recon_layers)

        if recon_layers is None:
            assert "embedding" in encoder_embed.keys()
            assert "log_concentration" in encoder_embed.keys()
            assert "reconstruction" in decoder_recon.keys()

        else:
            for lev in recon_layers:
                if lev != -1:
                    assert f"embedding_layer_{lev}" in encoder_embed.keys()
                    assert f"reconstruction_layer_{lev}" in decoder_recon.keys()

                else:
                    assert "embedding" in encoder_embed.keys()
                    assert "log_concentration" in encoder_embed.keys()
                    assert "reconstruction" in decoder_recon.keys()

        if recon_layers is None:
            assert encoder_embed["embedding"].shape[1] == ae_mnist_config.latent_dim
            assert encoder_embed["log_concentration"].shape[1] == 1
            assert (
                decoder_recon[f"reconstruction"].shape[1:] == ae_mnist_config.input_dim
            )

        else:
            if 1 in recon_layers:
                assert encoder_embed[f"embedding_layer_1"].shape[1] == 128
                assert decoder_recon[f"reconstruction_layer_1"].shape[1] == 1024

            if 2 in recon_layers:
                assert encoder_embed[f"embedding_layer_2"].shape[1] == 256
                assert decoder_recon[f"reconstruction_layer_2"].shape[1] == 512

            if 3 in recon_layers:
                assert encoder_embed[f"embedding_layer_3"].shape[1] == 512
                assert decoder_recon[f"reconstruction_layer_3"].shape[1] == 256

            if 4 in recon_layers:
                assert encoder_embed[f"embedding_layer_4"].shape[1] == 1024
                assert (
                    decoder_recon[f"reconstruction_layer_4"].shape[1:]
                    == ae_mnist_config.input_dim
                )

            if -1 in recon_layers:
                assert encoder_embed["embedding"].shape[1] == ae_mnist_config.latent_dim
                assert encoder_embed["log_concentration"].shape[1] == 1
                assert (
                    decoder_recon[f"reconstruction"].shape[1:]
                    == ae_mnist_config.input_dim
                )

    def test_discriminator(self, ae_mnist_config, mnist_like_data, recon_layers):

        ae_mnist_config.discriminator_input_dim = (1, 28, 28)

        discriminator = Discriminator_Conv_MNIST(ae_mnist_config).to(device)

        scores = discriminator(mnist_like_data, output_layer_levels=recon_layers)

        if recon_layers is None:
            assert "embedding" in scores.keys()

        else:
            for lev in recon_layers:
                if lev != -1:
                    assert f"embedding_layer_{lev}" in scores.keys()

                else:
                    assert "embedding" in scores.keys()

        if recon_layers is None:
            assert scores["embedding"].shape[1] == 1

        else:
            if 1 in recon_layers:
                assert scores[f"embedding_layer_1"].shape[1] == 128

            if 2 in recon_layers:
                assert scores[f"embedding_layer_2"].shape[1] == 256

            if 3 in recon_layers:
                assert scores[f"embedding_layer_3"].shape[1] == 512

            if 4 in recon_layers:
                assert scores[f"embedding_layer_4"].shape[1] == 1024

            if -1 in recon_layers:
                assert scores["embedding"].shape[1] == 1


class Test_MNIST_ResNets:
    @pytest.fixture(params=[[3, 4], [np.random.randint(1, 5)], [1, 2, 4, -1], None])
    def recon_layers(self, request):
        return request.param

    def test_ae_encoding_decoding(self, ae_mnist_config, mnist_like_data, recon_layers):
        encoder = Encoder_ResNet_AE_MNIST(ae_mnist_config).to(device)
        decoder = Decoder_ResNet_AE_MNIST(ae_mnist_config).to(device)

        embedding = encoder(mnist_like_data).embedding

        assert embedding.shape == (mnist_like_data.shape[0], ae_mnist_config.latent_dim)

        reconstruction = decoder(embedding).reconstruction

        assert reconstruction.shape == mnist_like_data.shape

        encoder_embed = encoder(mnist_like_data, output_layer_levels=recon_layers)
        decoder_recon = decoder(embedding, output_layer_levels=recon_layers)

        if recon_layers is None:
            assert "embedding" in encoder_embed.keys()
            assert "reconstruction" in decoder_recon.keys()

        else:
            for lev in recon_layers:
                if lev != -1:
                    assert f"embedding_layer_{lev}" in encoder_embed.keys()
                    assert f"reconstruction_layer_{lev}" in decoder_recon.keys()

                else:
                    assert "embedding" in encoder_embed.keys()
                    assert "reconstruction" in decoder_recon.keys()

        if recon_layers is None:
            assert encoder_embed["embedding"].shape[1] == ae_mnist_config.latent_dim
            assert (
                decoder_recon[f"reconstruction"].shape[1:] == ae_mnist_config.input_dim
            )

        else:
            if 1 in recon_layers:
                assert encoder_embed[f"embedding_layer_1"].shape[1] == 64
                assert decoder_recon[f"reconstruction_layer_1"].shape[1] == 128

            if 2 in recon_layers:
                assert encoder_embed[f"embedding_layer_2"].shape[1] == 128
                assert decoder_recon[f"reconstruction_layer_2"].shape[1] == 128

            if 3 in recon_layers:
                assert encoder_embed[f"embedding_layer_3"].shape[1] == 128
                assert decoder_recon[f"reconstruction_layer_3"].shape[1] == 128

            if 4 in recon_layers:
                assert encoder_embed[f"embedding_layer_4"].shape[1] == 128
                assert decoder_recon[f"reconstruction_layer_4"].shape[1] == 64

            if 5 in recon_layers:
                assert (
                    decoder_recon[f"reconstruction_layer_5"].shape[1:]
                    == ae_mnist_config.input_dim
                )

            if -1 in recon_layers:
                assert encoder_embed["embedding"].shape[1] == ae_mnist_config.latent_dim
                assert (
                    decoder_recon[f"reconstruction"].shape[1:]
                    == ae_mnist_config.input_dim
                )

    def test_vae_encoding_decoding(
        self, ae_mnist_config, mnist_like_data, recon_layers
    ):
        encoder = Encoder_ResNet_VAE_MNIST(ae_mnist_config).to(device)
        decoder = Decoder_ResNet_AE_MNIST(ae_mnist_config).to(device)

        embedding = encoder(mnist_like_data).embedding

        assert embedding.shape == (mnist_like_data.shape[0], ae_mnist_config.latent_dim)

        reconstruction = decoder(embedding).reconstruction

        assert reconstruction.shape == mnist_like_data.shape

        encoder_embed = encoder(mnist_like_data, output_layer_levels=recon_layers)
        decoder_recon = decoder(embedding, output_layer_levels=recon_layers)

        if recon_layers is None:
            assert "embedding" in encoder_embed.keys()
            assert "log_covariance" in encoder_embed.keys()
            assert "reconstruction" in decoder_recon.keys()

        else:
            for lev in recon_layers:
                if lev != -1:
                    assert f"embedding_layer_{lev}" in encoder_embed.keys()
                    assert f"reconstruction_layer_{lev}" in decoder_recon.keys()

                else:
                    assert "embedding" in encoder_embed.keys()
                    assert "log_covariance" in encoder_embed.keys()
                    assert "reconstruction" in decoder_recon.keys()

        if recon_layers is None:
            assert encoder_embed["embedding"].shape[1] == ae_mnist_config.latent_dim
            assert (
                encoder_embed["log_covariance"].shape[1] == ae_mnist_config.latent_dim
            )
            assert (
                decoder_recon[f"reconstruction"].shape[1:] == ae_mnist_config.input_dim
            )

        else:
            if 1 in recon_layers:
                assert encoder_embed[f"embedding_layer_1"].shape[1] == 64
                assert decoder_recon[f"reconstruction_layer_1"].shape[1] == 128

            if 2 in recon_layers:
                assert encoder_embed[f"embedding_layer_2"].shape[1] == 128
                assert decoder_recon[f"reconstruction_layer_2"].shape[1] == 128

            if 3 in recon_layers:
                assert encoder_embed[f"embedding_layer_3"].shape[1] == 128
                assert decoder_recon[f"reconstruction_layer_3"].shape[1] == 128

            if 4 in recon_layers:
                assert encoder_embed[f"embedding_layer_4"].shape[1] == 128
                assert decoder_recon[f"reconstruction_layer_4"].shape[1] == 64

            if 5 in recon_layers:
                assert (
                    decoder_recon[f"reconstruction_layer_5"].shape[1:]
                    == ae_mnist_config.input_dim
                )

            if -1 in recon_layers:
                assert encoder_embed["embedding"].shape[1] == ae_mnist_config.latent_dim
                assert (
                    encoder_embed["log_covariance"].shape[1]
                    == ae_mnist_config.latent_dim
                )
                assert (
                    decoder_recon[f"reconstruction"].shape[1:]
                    == ae_mnist_config.input_dim
                )

    def test_svae_encoding_decoding(
        self, ae_mnist_config, mnist_like_data, recon_layers
    ):
        encoder = Encoder_ResNet_SVAE_MNIST(ae_mnist_config).to(device)
        decoder = Decoder_ResNet_AE_MNIST(ae_mnist_config).to(device)

        embedding = encoder(mnist_like_data).embedding

        assert embedding.shape == (mnist_like_data.shape[0], ae_mnist_config.latent_dim)

        reconstruction = decoder(embedding).reconstruction

        assert reconstruction.shape == mnist_like_data.shape

        encoder_embed = encoder(mnist_like_data, output_layer_levels=recon_layers)
        decoder_recon = decoder(embedding, output_layer_levels=recon_layers)

        if recon_layers is None:
            assert "embedding" in encoder_embed.keys()
            assert "log_concentration" in encoder_embed.keys()
            assert "reconstruction" in decoder_recon.keys()

        else:
            for lev in recon_layers:
                if lev != -1:
                    assert f"embedding_layer_{lev}" in encoder_embed.keys()
                    assert f"reconstruction_layer_{lev}" in decoder_recon.keys()

                else:
                    assert "embedding" in encoder_embed.keys()
                    assert "log_concentration" in encoder_embed.keys()
                    assert "reconstruction" in decoder_recon.keys()

        if recon_layers is None:
            assert encoder_embed["embedding"].shape[1] == ae_mnist_config.latent_dim
            assert encoder_embed["log_concentration"].shape[1] == 1
            assert (
                decoder_recon[f"reconstruction"].shape[1:] == ae_mnist_config.input_dim
            )

        else:
            if 1 in recon_layers:
                assert encoder_embed[f"embedding_layer_1"].shape[1] == 64
                assert decoder_recon[f"reconstruction_layer_1"].shape[1] == 128

            if 2 in recon_layers:
                assert encoder_embed[f"embedding_layer_2"].shape[1] == 128
                assert decoder_recon[f"reconstruction_layer_2"].shape[1] == 128

            if 3 in recon_layers:
                assert encoder_embed[f"embedding_layer_3"].shape[1] == 128
                assert decoder_recon[f"reconstruction_layer_3"].shape[1] == 128

            if 4 in recon_layers:
                assert encoder_embed[f"embedding_layer_4"].shape[1] == 128
                assert decoder_recon[f"reconstruction_layer_4"].shape[1] == 64

            if 5 in recon_layers:
                assert (
                    decoder_recon[f"reconstruction_layer_5"].shape[1:]
                    == ae_mnist_config.input_dim
                )

            if -1 in recon_layers:
                assert encoder_embed["embedding"].shape[1] == ae_mnist_config.latent_dim
                assert encoder_embed["log_concentration"].shape[1] == 1
                assert (
                    decoder_recon[f"reconstruction"].shape[1:]
                    == ae_mnist_config.input_dim
                )

    def test_vqvae_encoding_decoding(
        self, ae_mnist_config, mnist_like_data, recon_layers
    ):
        encoder = Encoder_ResNet_VQVAE_MNIST(ae_mnist_config).to(device)
        decoder = Decoder_ResNet_VQVAE_MNIST(ae_mnist_config).to(device)

        embedding = encoder(mnist_like_data).embedding

        assert embedding.shape[:2] == (
            mnist_like_data.shape[0],
            ae_mnist_config.latent_dim,
        )

        reconstruction = decoder(embedding).reconstruction

        assert reconstruction.shape == mnist_like_data.shape

        encoder_embed = encoder(mnist_like_data, output_layer_levels=recon_layers)
        decoder_recon = decoder(embedding, output_layer_levels=recon_layers)

        if recon_layers is None:
            assert "embedding" in encoder_embed.keys()
            assert "reconstruction" in decoder_recon.keys()

        else:
            for lev in recon_layers:
                if lev != -1:
                    assert f"embedding_layer_{lev}" in encoder_embed.keys()
                    assert f"reconstruction_layer_{lev}" in decoder_recon.keys()

                else:
                    assert "embedding" in encoder_embed.keys()
                    assert "reconstruction" in decoder_recon.keys()

        if recon_layers is None:
            assert encoder_embed["embedding"].shape[1] == ae_mnist_config.latent_dim
            assert (
                decoder_recon[f"reconstruction"].shape[1:] == ae_mnist_config.input_dim
            )

        else:
            if 1 in recon_layers:
                assert encoder_embed[f"embedding_layer_1"].shape[1] == 64
                assert decoder_recon[f"reconstruction_layer_1"].shape[1] == 128

            if 2 in recon_layers:
                assert encoder_embed[f"embedding_layer_2"].shape[1] == 128
                assert decoder_recon[f"reconstruction_layer_2"].shape[1] == 128

            if 3 in recon_layers:
                assert encoder_embed[f"embedding_layer_3"].shape[1] == 128
                assert decoder_recon[f"reconstruction_layer_3"].shape[1] == 128

            if 4 in recon_layers:
                assert encoder_embed[f"embedding_layer_4"].shape[1] == 128
                assert decoder_recon[f"reconstruction_layer_4"].shape[1] == 64

            if 5 in recon_layers:
                assert (
                    decoder_recon[f"reconstruction_layer_5"].shape[1:]
                    == ae_mnist_config.input_dim
                )

            if -1 in recon_layers:
                assert encoder_embed["embedding"].shape[1] == ae_mnist_config.latent_dim
                assert (
                    decoder_recon[f"reconstruction"].shape[1:]
                    == ae_mnist_config.input_dim
                )


class Test_CIFAR_ConvNets:
    @pytest.fixture(params=[[3, 4], [np.random.randint(1, 5)], [1, 2, 4, -1], None])
    def recon_layers(self, request):
        return request.param

    def test_ae_encoding_decoding(self, ae_cifar_config, cifar_like_data, recon_layers):
        encoder = Encoder_Conv_AE_CIFAR(ae_cifar_config).to(device)
        decoder = Decoder_Conv_AE_CIFAR(ae_cifar_config).to(device)

        embedding = encoder(cifar_like_data).embedding

        assert embedding.shape == (cifar_like_data.shape[0], ae_cifar_config.latent_dim)

        reconstruction = decoder(embedding).reconstruction

        assert reconstruction.shape == cifar_like_data.shape

        encoder_embed = encoder(cifar_like_data, output_layer_levels=recon_layers)
        decoder_recon = decoder(embedding, output_layer_levels=recon_layers)

        if recon_layers is None:
            assert "embedding" in encoder_embed.keys()
            assert "reconstruction" in decoder_recon.keys()

        else:
            for lev in recon_layers:
                if lev != -1:
                    assert f"embedding_layer_{lev}" in encoder_embed.keys()
                    assert f"reconstruction_layer_{lev}" in decoder_recon.keys()

                else:
                    assert "embedding" in encoder_embed.keys()
                    assert "reconstruction" in decoder_recon.keys()

        if recon_layers is None:
            assert encoder_embed["embedding"].shape[1] == ae_cifar_config.latent_dim
            assert (
                decoder_recon[f"reconstruction"].shape[1:] == ae_cifar_config.input_dim
            )

        else:
            if 1 in recon_layers:
                assert encoder_embed[f"embedding_layer_1"].shape[1] == 128
                assert decoder_recon[f"reconstruction_layer_1"].shape[1] == 1024

            if 2 in recon_layers:
                assert encoder_embed[f"embedding_layer_2"].shape[1] == 256
                assert decoder_recon[f"reconstruction_layer_2"].shape[1] == 512

            if 3 in recon_layers:
                assert encoder_embed[f"embedding_layer_3"].shape[1] == 512
                assert decoder_recon[f"reconstruction_layer_3"].shape[1] == 256

            if 4 in recon_layers:
                assert encoder_embed[f"embedding_layer_4"].shape[1] == 1024
                assert (
                    decoder_recon[f"reconstruction_layer_4"].shape[1:]
                    == ae_cifar_config.input_dim
                )

            if -1 in recon_layers:
                assert encoder_embed["embedding"].shape[1] == ae_cifar_config.latent_dim
                assert (
                    decoder_recon[f"reconstruction"].shape[1:]
                    == ae_cifar_config.input_dim
                )

    def test_vae_encoding_decoding(
        self, ae_cifar_config, cifar_like_data, recon_layers
    ):
        encoder = Encoder_Conv_VAE_CIFAR(ae_cifar_config).to(device)
        decoder = Decoder_Conv_AE_CIFAR(ae_cifar_config).to(device)

        output = encoder(cifar_like_data)
        embedding, log_covariance = output.embedding, output.log_covariance

        assert embedding.shape == (
            cifar_like_data.shape[0],
            ae_cifar_config.latent_dim,
        )
        assert log_covariance.shape == (
            cifar_like_data.shape[0],
            ae_cifar_config.latent_dim,
        )

        reconstruction = decoder(embedding).reconstruction

        assert reconstruction.shape == cifar_like_data.shape

        encoder_embed = encoder(cifar_like_data, output_layer_levels=recon_layers)
        decoder_recon = decoder(embedding, output_layer_levels=recon_layers)

        if recon_layers is None:
            assert "embedding" in encoder_embed.keys()
            assert "log_covariance" in encoder_embed.keys()
            assert "reconstruction" in decoder_recon.keys()

        else:
            for lev in recon_layers:
                if lev != -1:
                    assert f"embedding_layer_{lev}" in encoder_embed.keys()
                    assert f"reconstruction_layer_{lev}" in decoder_recon.keys()

                else:
                    assert "embedding" in encoder_embed.keys()
                    assert "log_covariance" in encoder_embed.keys()
                    assert "reconstruction" in decoder_recon.keys()

        if recon_layers is None:
            assert encoder_embed["embedding"].shape[1] == ae_cifar_config.latent_dim
            assert (
                encoder_embed["log_covariance"].shape[1] == ae_cifar_config.latent_dim
            )
            assert (
                decoder_recon[f"reconstruction"].shape[1:] == ae_cifar_config.input_dim
            )

        else:
            if 1 in recon_layers:
                assert encoder_embed[f"embedding_layer_1"].shape[1] == 128
                assert decoder_recon[f"reconstruction_layer_1"].shape[1] == 1024

            if 2 in recon_layers:
                assert encoder_embed[f"embedding_layer_2"].shape[1] == 256
                assert decoder_recon[f"reconstruction_layer_2"].shape[1] == 512

            if 3 in recon_layers:
                assert encoder_embed[f"embedding_layer_3"].shape[1] == 512
                assert decoder_recon[f"reconstruction_layer_3"].shape[1] == 256

            if 4 in recon_layers:
                assert encoder_embed[f"embedding_layer_4"].shape[1] == 1024
                assert (
                    decoder_recon[f"reconstruction_layer_4"].shape[1:]
                    == ae_cifar_config.input_dim
                )

            if -1 in recon_layers:
                assert encoder_embed["embedding"].shape[1] == ae_cifar_config.latent_dim
                assert (
                    encoder_embed["log_covariance"].shape[1]
                    == ae_cifar_config.latent_dim
                )
                assert (
                    decoder_recon[f"reconstruction"].shape[1:]
                    == ae_cifar_config.input_dim
                )

    def test_svae_encoding_decoding(
        self, ae_cifar_config, cifar_like_data, recon_layers
    ):
        encoder = Encoder_Conv_SVAE_CIFAR(ae_cifar_config).to(device)
        decoder = Decoder_Conv_AE_CIFAR(ae_cifar_config).to(device)

        output = encoder(cifar_like_data)
        embedding, log_concentration = output.embedding, output.log_concentration

        assert embedding.shape == (
            cifar_like_data.shape[0],
            ae_cifar_config.latent_dim,
        )
        assert log_concentration.shape == (
            cifar_like_data.shape[0],
            1,
        )

        reconstruction = decoder(embedding).reconstruction

        assert reconstruction.shape == cifar_like_data.shape

        encoder_embed = encoder(cifar_like_data, output_layer_levels=recon_layers)
        decoder_recon = decoder(embedding, output_layer_levels=recon_layers)

        if recon_layers is None:
            assert "embedding" in encoder_embed.keys()
            assert "log_concentration" in encoder_embed.keys()
            assert "reconstruction" in decoder_recon.keys()

        else:
            for lev in recon_layers:
                if lev != -1:
                    assert f"embedding_layer_{lev}" in encoder_embed.keys()
                    assert f"reconstruction_layer_{lev}" in decoder_recon.keys()

                else:
                    assert "embedding" in encoder_embed.keys()
                    assert "log_concentration" in encoder_embed.keys()
                    assert "reconstruction" in decoder_recon.keys()

        if recon_layers is None:
            assert encoder_embed["embedding"].shape[1] == ae_cifar_config.latent_dim
            assert encoder_embed["log_concentration"].shape[1] == 1
            assert (
                decoder_recon[f"reconstruction"].shape[1:] == ae_cifar_config.input_dim
            )

        else:
            if 1 in recon_layers:
                assert encoder_embed[f"embedding_layer_1"].shape[1] == 128
                assert decoder_recon[f"reconstruction_layer_1"].shape[1] == 1024

            if 2 in recon_layers:
                assert encoder_embed[f"embedding_layer_2"].shape[1] == 256
                assert decoder_recon[f"reconstruction_layer_2"].shape[1] == 512

            if 3 in recon_layers:
                assert encoder_embed[f"embedding_layer_3"].shape[1] == 512
                assert decoder_recon[f"reconstruction_layer_3"].shape[1] == 256

            if 4 in recon_layers:
                assert encoder_embed[f"embedding_layer_4"].shape[1] == 1024
                assert (
                    decoder_recon[f"reconstruction_layer_4"].shape[1:]
                    == ae_cifar_config.input_dim
                )

            if -1 in recon_layers:
                assert encoder_embed["embedding"].shape[1] == ae_cifar_config.latent_dim
                assert encoder_embed["log_concentration"].shape[1] == 1
                assert (
                    decoder_recon[f"reconstruction"].shape[1:]
                    == ae_cifar_config.input_dim
                )

    def test_discriminator(self, ae_cifar_config, cifar_like_data, recon_layers):

        ae_cifar_config.discriminator_input_dim = (3, 32, 32)

        discriminator = Discriminator_Conv_CIFAR(ae_cifar_config).to(device)

        scores = discriminator(cifar_like_data, output_layer_levels=recon_layers)

        if recon_layers is None:
            assert "embedding" in scores.keys()

        else:
            for lev in recon_layers:
                if lev != -1:
                    assert f"embedding_layer_{lev}" in scores.keys()

                else:
                    assert "embedding" in scores.keys()

        if recon_layers is None:
            assert scores["embedding"].shape[1] == 1

        else:
            if 1 in recon_layers:
                assert scores[f"embedding_layer_1"].shape[1] == 128

            if 2 in recon_layers:
                assert scores[f"embedding_layer_2"].shape[1] == 256

            if 3 in recon_layers:
                assert scores[f"embedding_layer_3"].shape[1] == 512

            if 4 in recon_layers:
                assert scores[f"embedding_layer_4"].shape[1] == 1024

            if -1 in recon_layers:
                assert scores["embedding"].shape[1] == 1


class Test_CIFAR_ResNets:
    @pytest.fixture(params=[[3, 4], [np.random.randint(1, 5)], [1, 2, 4, -1], None])
    def recon_layers(self, request):
        return request.param

    def test_ae_encoding_decoding(self, ae_cifar_config, cifar_like_data, recon_layers):
        encoder = Encoder_ResNet_AE_CIFAR(ae_cifar_config).to(device)
        decoder = Decoder_ResNet_AE_CIFAR(ae_cifar_config).to(device)

        embedding = encoder(cifar_like_data).embedding

        assert embedding.shape == (cifar_like_data.shape[0], ae_cifar_config.latent_dim)

        reconstruction = decoder(embedding).reconstruction

        assert reconstruction.shape == cifar_like_data.shape

        encoder_embed = encoder(cifar_like_data, output_layer_levels=recon_layers)
        decoder_recon = decoder(embedding, output_layer_levels=recon_layers)

        if recon_layers is None:
            assert "embedding" in encoder_embed.keys()
            assert "reconstruction" in decoder_recon.keys()

        else:
            for lev in recon_layers:
                if lev != -1:
                    assert f"embedding_layer_{lev}" in encoder_embed.keys()
                    assert f"reconstruction_layer_{lev}" in decoder_recon.keys()

                else:
                    assert "embedding" in encoder_embed.keys()
                    assert "reconstruction" in decoder_recon.keys()

        if recon_layers is None:
            assert encoder_embed["embedding"].shape[1] == ae_cifar_config.latent_dim
            assert (
                decoder_recon[f"reconstruction"].shape[1:] == ae_cifar_config.input_dim
            )

        else:
            if 1 in recon_layers:
                assert encoder_embed[f"embedding_layer_1"].shape[1] == 64
                assert decoder_recon[f"reconstruction_layer_1"].shape[1] == 128

            if 2 in recon_layers:
                assert encoder_embed[f"embedding_layer_2"].shape[1] == 128
                assert decoder_recon[f"reconstruction_layer_2"].shape[1] == 128

            if 3 in recon_layers:
                assert encoder_embed[f"embedding_layer_3"].shape[1] == 128
                assert decoder_recon[f"reconstruction_layer_3"].shape[1] == 64

            if 4 in recon_layers:
                assert encoder_embed[f"embedding_layer_4"].shape[1] == 128
                assert (
                    decoder_recon[f"reconstruction_layer_4"].shape[1:]
                    == ae_cifar_config.input_dim
                )

            if -1 in recon_layers:
                assert encoder_embed["embedding"].shape[1] == ae_cifar_config.latent_dim
                assert (
                    decoder_recon[f"reconstruction"].shape[1:]
                    == ae_cifar_config.input_dim
                )

    def test_vae_encoding_decoding(
        self, ae_cifar_config, cifar_like_data, recon_layers
    ):
        encoder = Encoder_ResNet_VAE_CIFAR(ae_cifar_config).to(device)
        decoder = Decoder_ResNet_AE_CIFAR(ae_cifar_config).to(device)

        embedding = encoder(cifar_like_data).embedding

        assert embedding.shape == (cifar_like_data.shape[0], ae_cifar_config.latent_dim)

        reconstruction = decoder(embedding).reconstruction

        assert reconstruction.shape == cifar_like_data.shape

        encoder_embed = encoder(cifar_like_data, output_layer_levels=recon_layers)
        decoder_recon = decoder(embedding, output_layer_levels=recon_layers)

        if recon_layers is None:
            assert "embedding" in encoder_embed.keys()
            assert "log_covariance" in encoder_embed.keys()
            assert "reconstruction" in decoder_recon.keys()

        else:
            for lev in recon_layers:
                if lev != -1:
                    assert f"embedding_layer_{lev}" in encoder_embed.keys()
                    assert f"reconstruction_layer_{lev}" in decoder_recon.keys()

                else:
                    assert "embedding" in encoder_embed.keys()
                    assert "log_covariance" in encoder_embed.keys()
                    assert "reconstruction" in decoder_recon.keys()

        if recon_layers is None:
            assert encoder_embed["embedding"].shape[1] == ae_cifar_config.latent_dim
            assert (
                encoder_embed["log_covariance"].shape[1] == ae_cifar_config.latent_dim
            )
            assert (
                decoder_recon[f"reconstruction"].shape[1:] == ae_cifar_config.input_dim
            )

        else:
            if 1 in recon_layers:
                assert encoder_embed[f"embedding_layer_1"].shape[1] == 64
                assert decoder_recon[f"reconstruction_layer_1"].shape[1] == 128

            if 2 in recon_layers:
                assert encoder_embed[f"embedding_layer_2"].shape[1] == 128
                assert decoder_recon[f"reconstruction_layer_2"].shape[1] == 128

            if 3 in recon_layers:
                assert encoder_embed[f"embedding_layer_3"].shape[1] == 128
                assert decoder_recon[f"reconstruction_layer_3"].shape[1] == 64

            if 4 in recon_layers:
                assert encoder_embed[f"embedding_layer_4"].shape[1] == 128
                assert (
                    decoder_recon[f"reconstruction_layer_4"].shape[1:]
                    == ae_cifar_config.input_dim
                )

            if -1 in recon_layers:
                assert encoder_embed["embedding"].shape[1] == ae_cifar_config.latent_dim
                assert (
                    encoder_embed["log_covariance"].shape[1]
                    == ae_cifar_config.latent_dim
                )
                assert (
                    decoder_recon[f"reconstruction"].shape[1:]
                    == ae_cifar_config.input_dim
                )

    def test_svae_encoding_decoding(
        self, ae_cifar_config, cifar_like_data, recon_layers
    ):
        encoder = Encoder_ResNet_SVAE_CIFAR(ae_cifar_config).to(device)
        decoder = Decoder_ResNet_AE_CIFAR(ae_cifar_config).to(device)

        embedding = encoder(cifar_like_data).embedding

        assert embedding.shape == (cifar_like_data.shape[0], ae_cifar_config.latent_dim)

        reconstruction = decoder(embedding).reconstruction

        assert reconstruction.shape == cifar_like_data.shape

        encoder_embed = encoder(cifar_like_data, output_layer_levels=recon_layers)
        decoder_recon = decoder(embedding, output_layer_levels=recon_layers)

        if recon_layers is None:
            assert "embedding" in encoder_embed.keys()
            assert "log_concentration" in encoder_embed.keys()
            assert "reconstruction" in decoder_recon.keys()

        else:
            for lev in recon_layers:
                if lev != -1:
                    assert f"embedding_layer_{lev}" in encoder_embed.keys()
                    assert f"reconstruction_layer_{lev}" in decoder_recon.keys()

                else:
                    assert "embedding" in encoder_embed.keys()
                    assert "log_concentration" in encoder_embed.keys()
                    assert "reconstruction" in decoder_recon.keys()

        if recon_layers is None:
            assert encoder_embed["embedding"].shape[1] == ae_cifar_config.latent_dim
            assert encoder_embed["log_concentration"].shape[1] == 1
            assert (
                decoder_recon[f"reconstruction"].shape[1:] == ae_cifar_config.input_dim
            )

        else:
            if 1 in recon_layers:
                assert encoder_embed[f"embedding_layer_1"].shape[1] == 64
                assert decoder_recon[f"reconstruction_layer_1"].shape[1] == 128

            if 2 in recon_layers:
                assert encoder_embed[f"embedding_layer_2"].shape[1] == 128
                assert decoder_recon[f"reconstruction_layer_2"].shape[1] == 128

            if 3 in recon_layers:
                assert encoder_embed[f"embedding_layer_3"].shape[1] == 128
                assert decoder_recon[f"reconstruction_layer_3"].shape[1] == 64

            if 4 in recon_layers:
                assert encoder_embed[f"embedding_layer_4"].shape[1] == 128
                assert (
                    decoder_recon[f"reconstruction_layer_4"].shape[1:]
                    == ae_cifar_config.input_dim
                )

            if -1 in recon_layers:
                assert encoder_embed["embedding"].shape[1] == ae_cifar_config.latent_dim
                assert encoder_embed["log_concentration"].shape[1] == 1
                assert (
                    decoder_recon[f"reconstruction"].shape[1:]
                    == ae_cifar_config.input_dim
                )

    def test_vqvae_encoding_decoding(
        self, ae_cifar_config, cifar_like_data, recon_layers
    ):
        encoder = Encoder_ResNet_VQVAE_CIFAR(ae_cifar_config).to(device)
        decoder = Decoder_ResNet_VQVAE_CIFAR(ae_cifar_config).to(device)

        embedding = encoder(cifar_like_data).embedding

        assert embedding.shape[:2] == (
            cifar_like_data.shape[0],
            ae_cifar_config.latent_dim,
        )

        reconstruction = decoder(embedding).reconstruction

        assert reconstruction.shape == cifar_like_data.shape

        encoder_embed = encoder(cifar_like_data, output_layer_levels=recon_layers)
        decoder_recon = decoder(embedding, output_layer_levels=recon_layers)

        if recon_layers is None:
            assert "embedding" in encoder_embed.keys()
            assert "reconstruction" in decoder_recon.keys()

        else:
            for lev in recon_layers:
                if lev != -1:
                    assert f"embedding_layer_{lev}" in encoder_embed.keys()
                    assert f"reconstruction_layer_{lev}" in decoder_recon.keys()

                else:
                    assert "embedding" in encoder_embed.keys()
                    assert "reconstruction" in decoder_recon.keys()

        if recon_layers is None:
            assert encoder_embed["embedding"].shape[1] == ae_cifar_config.latent_dim
            assert (
                decoder_recon[f"reconstruction"].shape[1:] == ae_cifar_config.input_dim
            )

        else:
            if 1 in recon_layers:
                assert encoder_embed[f"embedding_layer_1"].shape[1] == 64
                assert decoder_recon[f"reconstruction_layer_1"].shape[1] == 128

            if 2 in recon_layers:
                assert encoder_embed[f"embedding_layer_2"].shape[1] == 128
                assert decoder_recon[f"reconstruction_layer_2"].shape[1] == 128

            if 3 in recon_layers:
                assert encoder_embed[f"embedding_layer_3"].shape[1] == 128
                assert decoder_recon[f"reconstruction_layer_3"].shape[1] == 64

            if 4 in recon_layers:
                assert encoder_embed[f"embedding_layer_4"].shape[1] == 128
                assert (
                    decoder_recon[f"reconstruction_layer_4"].shape[1:]
                    == ae_cifar_config.input_dim
                )

            if -1 in recon_layers:
                assert encoder_embed["embedding"].shape[1] == ae_cifar_config.latent_dim
                assert (
                    decoder_recon[f"reconstruction"].shape[1:]
                    == ae_cifar_config.input_dim
                )


class Test_CELEBA_ConvNets:
    @pytest.fixture(params=[[3, 4], [np.random.randint(1, 5)], [1, 2, 4, -1], None])
    def recon_layers(self, request):
        return request.param

    def test_ae_encoding_decoding(
        self, ae_celeba_config, celeba_like_data, recon_layers
    ):
        encoder = Encoder_Conv_AE_CELEBA(ae_celeba_config).to(device)
        decoder = Decoder_Conv_AE_CELEBA(ae_celeba_config).to(device)

        embedding = encoder(celeba_like_data).embedding

        assert embedding.shape == (
            celeba_like_data.shape[0],
            ae_celeba_config.latent_dim,
        )

        reconstruction = decoder(embedding).reconstruction

        assert reconstruction.shape == celeba_like_data.shape

        encoder_embed = encoder(celeba_like_data, output_layer_levels=recon_layers)
        decoder_recon = decoder(embedding, output_layer_levels=recon_layers)

        if recon_layers is None:
            assert "embedding" in encoder_embed.keys()
            assert "reconstruction" in decoder_recon.keys()

        else:
            for lev in recon_layers:
                if lev != -1:
                    assert f"embedding_layer_{lev}" in encoder_embed.keys()
                    assert f"reconstruction_layer_{lev}" in decoder_recon.keys()

                else:
                    assert "embedding" in encoder_embed.keys()
                    assert "reconstruction" in decoder_recon.keys()

        if recon_layers is None:
            assert encoder_embed["embedding"].shape[1] == ae_celeba_config.latent_dim
            assert (
                decoder_recon[f"reconstruction"].shape[1:] == ae_celeba_config.input_dim
            )

        else:
            if 1 in recon_layers:
                assert encoder_embed[f"embedding_layer_1"].shape[1] == 128
                assert decoder_recon[f"reconstruction_layer_1"].shape[1] == 1024

            if 2 in recon_layers:
                assert encoder_embed[f"embedding_layer_2"].shape[1] == 256
                assert decoder_recon[f"reconstruction_layer_2"].shape[1] == 512

            if 3 in recon_layers:
                assert encoder_embed[f"embedding_layer_3"].shape[1] == 512
                assert decoder_recon[f"reconstruction_layer_3"].shape[1] == 256

            if 4 in recon_layers:
                assert encoder_embed[f"embedding_layer_4"].shape[1] == 1024
                assert decoder_recon[f"reconstruction_layer_4"].shape[1] == 128

            if 5 in recon_layers:
                assert (
                    decoder_recon[f"reconstruction_layer_4"].shape[1:]
                    == ae_celeba_config.input_dim
                )

            if -1 in recon_layers:
                assert (
                    encoder_embed["embedding"].shape[1] == ae_celeba_config.latent_dim
                )
                assert (
                    decoder_recon[f"reconstruction"].shape[1:]
                    == ae_celeba_config.input_dim
                )

    def test_vae_encoding_decoding(
        self, ae_celeba_config, celeba_like_data, recon_layers
    ):
        encoder = Encoder_Conv_VAE_CELEBA(ae_celeba_config).to(device)
        decoder = Decoder_Conv_AE_CELEBA(ae_celeba_config).to(device)

        output = encoder(celeba_like_data)
        embedding, log_covariance = output.embedding, output.log_covariance

        assert embedding.shape == (
            celeba_like_data.shape[0],
            ae_celeba_config.latent_dim,
        )
        assert log_covariance.shape == (
            celeba_like_data.shape[0],
            ae_celeba_config.latent_dim,
        )

        reconstruction = decoder(embedding).reconstruction

        assert reconstruction.shape == celeba_like_data.shape

        encoder_embed = encoder(celeba_like_data, output_layer_levels=recon_layers)
        decoder_recon = decoder(embedding, output_layer_levels=recon_layers)

        if recon_layers is None:
            assert "embedding" in encoder_embed.keys()
            assert "log_covariance" in encoder_embed.keys()
            assert "reconstruction" in decoder_recon.keys()

        else:
            for lev in recon_layers:
                if lev != -1:
                    assert f"embedding_layer_{lev}" in encoder_embed.keys()
                    assert f"reconstruction_layer_{lev}" in decoder_recon.keys()

                else:
                    assert "embedding" in encoder_embed.keys()
                    assert "log_covariance" in encoder_embed.keys()
                    assert "reconstruction" in decoder_recon.keys()

        if recon_layers is None:
            assert encoder_embed["embedding"].shape[1] == ae_celeba_config.latent_dim
            assert (
                encoder_embed["log_covariance"].shape[1] == ae_celeba_config.latent_dim
            )
            assert (
                decoder_recon[f"reconstruction"].shape[1:] == ae_celeba_config.input_dim
            )

        else:
            if 1 in recon_layers:
                assert encoder_embed[f"embedding_layer_1"].shape[1] == 128
                assert decoder_recon[f"reconstruction_layer_1"].shape[1] == 1024

            if 2 in recon_layers:
                assert encoder_embed[f"embedding_layer_2"].shape[1] == 256
                assert decoder_recon[f"reconstruction_layer_2"].shape[1] == 512

            if 3 in recon_layers:
                assert encoder_embed[f"embedding_layer_3"].shape[1] == 512
                assert decoder_recon[f"reconstruction_layer_3"].shape[1] == 256

            if 4 in recon_layers:
                assert encoder_embed[f"embedding_layer_4"].shape[1] == 1024
                assert decoder_recon[f"reconstruction_layer_4"].shape[1] == 128

            if 5 in recon_layers:
                assert (
                    decoder_recon[f"reconstruction_layer_4"].shape[1:]
                    == ae_celeba_config.input_dim
                )

            if -1 in recon_layers:
                assert (
                    encoder_embed["embedding"].shape[1] == ae_celeba_config.latent_dim
                )
                assert (
                    encoder_embed["log_covariance"].shape[1]
                    == ae_celeba_config.latent_dim
                )
                assert (
                    decoder_recon[f"reconstruction"].shape[1:]
                    == ae_celeba_config.input_dim
                )

    def test_svae_encoding_decoding(
        self, ae_celeba_config, celeba_like_data, recon_layers
    ):
        encoder = Encoder_Conv_SVAE_CELEBA(ae_celeba_config).to(device)
        decoder = Decoder_Conv_AE_CELEBA(ae_celeba_config).to(device)

        output = encoder(celeba_like_data)
        embedding, log_concentration = output.embedding, output.log_concentration

        assert embedding.shape == (
            celeba_like_data.shape[0],
            ae_celeba_config.latent_dim,
        )
        assert log_concentration.shape == (celeba_like_data.shape[0], 1)

        reconstruction = decoder(embedding).reconstruction

        assert reconstruction.shape == celeba_like_data.shape

        encoder_embed = encoder(celeba_like_data, output_layer_levels=recon_layers)
        decoder_recon = decoder(embedding, output_layer_levels=recon_layers)

        if recon_layers is None:
            assert "embedding" in encoder_embed.keys()
            assert "log_concentration" in encoder_embed.keys()
            assert "reconstruction" in decoder_recon.keys()

        else:
            for lev in recon_layers:
                if lev != -1:
                    assert f"embedding_layer_{lev}" in encoder_embed.keys()
                    assert f"reconstruction_layer_{lev}" in decoder_recon.keys()

                else:
                    assert "embedding" in encoder_embed.keys()
                    assert "log_concentration" in encoder_embed.keys()
                    assert "reconstruction" in decoder_recon.keys()

        if recon_layers is None:
            assert encoder_embed["embedding"].shape[1] == ae_celeba_config.latent_dim
            assert encoder_embed["log_concentration"].shape[1] == 1
            assert (
                decoder_recon[f"reconstruction"].shape[1:] == ae_celeba_config.input_dim
            )

        else:
            if 1 in recon_layers:
                assert encoder_embed[f"embedding_layer_1"].shape[1] == 128
                assert decoder_recon[f"reconstruction_layer_1"].shape[1] == 1024

            if 2 in recon_layers:
                assert encoder_embed[f"embedding_layer_2"].shape[1] == 256
                assert decoder_recon[f"reconstruction_layer_2"].shape[1] == 512

            if 3 in recon_layers:
                assert encoder_embed[f"embedding_layer_3"].shape[1] == 512
                assert decoder_recon[f"reconstruction_layer_3"].shape[1] == 256

            if 4 in recon_layers:
                assert encoder_embed[f"embedding_layer_4"].shape[1] == 1024
                assert decoder_recon[f"reconstruction_layer_4"].shape[1] == 128

            if 5 in recon_layers:
                assert (
                    decoder_recon[f"reconstruction_layer_4"].shape[1:]
                    == ae_celeba_config.input_dim
                )

            if -1 in recon_layers:
                assert (
                    encoder_embed["embedding"].shape[1] == ae_celeba_config.latent_dim
                )
                assert encoder_embed["log_concentration"].shape[1] == 1
                assert (
                    decoder_recon[f"reconstruction"].shape[1:]
                    == ae_celeba_config.input_dim
                )

    def test_discriminator(self, ae_celeba_config, celeba_like_data, recon_layers):

        ae_celeba_config.discriminator_input_dim = (3, 64, 64)

        discriminator = Discriminator_Conv_CELEBA(ae_celeba_config).to(device)

        scores = discriminator(celeba_like_data, output_layer_levels=recon_layers)

        if recon_layers is None:
            assert "embedding" in scores.keys()

        else:
            for lev in recon_layers:
                if lev != -1:
                    assert f"embedding_layer_{lev}" in scores.keys()

                else:
                    assert "embedding" in scores.keys()

        if recon_layers is None:
            assert scores["embedding"].shape[1] == 1

        else:
            if 1 in recon_layers:
                assert scores[f"embedding_layer_1"].shape[1] == 128

            if 2 in recon_layers:
                assert scores[f"embedding_layer_2"].shape[1] == 256

            if 3 in recon_layers:
                assert scores[f"embedding_layer_3"].shape[1] == 512

            if 4 in recon_layers:
                assert scores[f"embedding_layer_4"].shape[1] == 1024

            if 5 in recon_layers:
                assert scores[f"embedding_layer_5"].shape[1] == 1

            if -1 in recon_layers:
                assert scores["embedding"].shape[1] == 1


class Test_CELEBA_ResNets:
    @pytest.fixture(params=[[3, 4], [np.random.randint(1, 6)], [1, 2, 4, -1], None])
    def recon_layers(self, request):
        return request.param

    def test_ae_encoding_decoding(
        self, ae_celeba_config, celeba_like_data, recon_layers
    ):
        encoder = Encoder_ResNet_AE_CELEBA(ae_celeba_config).to(device)
        decoder = Decoder_ResNet_AE_CELEBA(ae_celeba_config).to(device)

        embedding = encoder(celeba_like_data).embedding

        assert embedding.shape == (
            celeba_like_data.shape[0],
            ae_celeba_config.latent_dim,
        )

        reconstruction = decoder(embedding).reconstruction

        assert reconstruction.shape == celeba_like_data.shape

        encoder_embed = encoder(celeba_like_data, output_layer_levels=recon_layers)
        decoder_recon = decoder(embedding, output_layer_levels=recon_layers)

        if recon_layers is None:
            assert "embedding" in encoder_embed.keys()
            assert "reconstruction" in decoder_recon.keys()

        else:
            for lev in recon_layers:
                if lev != -1:
                    assert f"embedding_layer_{lev}" in encoder_embed.keys()
                    assert f"reconstruction_layer_{lev}" in decoder_recon.keys()

                else:
                    assert "embedding" in encoder_embed.keys()
                    assert "reconstruction" in decoder_recon.keys()

        if recon_layers is None:
            assert encoder_embed["embedding"].shape[1] == ae_celeba_config.latent_dim
            assert (
                decoder_recon[f"reconstruction"].shape[1:] == ae_celeba_config.input_dim
            )

        else:
            if 1 in recon_layers:
                assert encoder_embed[f"embedding_layer_1"].shape[1] == 64
                assert decoder_recon[f"reconstruction_layer_1"].shape[1] == 128

            if 2 in recon_layers:
                assert encoder_embed[f"embedding_layer_2"].shape[1] == 128
                assert decoder_recon[f"reconstruction_layer_2"].shape[1] == 128

            if 3 in recon_layers:
                assert encoder_embed[f"embedding_layer_3"].shape[1] == 128
                assert decoder_recon[f"reconstruction_layer_3"].shape[1] == 128

            if 4 in recon_layers:
                assert encoder_embed[f"embedding_layer_4"].shape[1] == 128
                assert decoder_recon[f"reconstruction_layer_4"].shape[1] == 128

            if 5 in recon_layers:
                assert encoder_embed[f"embedding_layer_5"].shape[1] == 128
                assert decoder_recon[f"reconstruction_layer_5"].shape[1] == 64

            if 6 in recon_layers:
                assert (
                    decoder_recon[f"reconstruction_layer_6"].shape[1:]
                    == ae_celeba_config.input_dim
                )

            if -1 in recon_layers:
                assert (
                    encoder_embed["embedding"].shape[1] == ae_celeba_config.latent_dim
                )
                assert (
                    decoder_recon[f"reconstruction"].shape[1:]
                    == ae_celeba_config.input_dim
                )

    def test_vae_encoding_decoding(
        self, ae_celeba_config, celeba_like_data, recon_layers
    ):
        encoder = Encoder_ResNet_VAE_CELEBA(ae_celeba_config).to(device)
        decoder = Decoder_ResNet_AE_CELEBA(ae_celeba_config).to(device)

        output = encoder(celeba_like_data)
        embedding, log_covariance = output.embedding, output.log_covariance

        assert embedding.shape == (
            celeba_like_data.shape[0],
            ae_celeba_config.latent_dim,
        )
        assert log_covariance.shape == (
            celeba_like_data.shape[0],
            ae_celeba_config.latent_dim,
        )

        reconstruction = decoder(embedding).reconstruction

        assert reconstruction.shape == celeba_like_data.shape

        encoder_embed = encoder(celeba_like_data, output_layer_levels=recon_layers)
        decoder_recon = decoder(embedding, output_layer_levels=recon_layers)

        if recon_layers is None:
            assert "embedding" in encoder_embed.keys()
            assert "log_covariance" in encoder_embed.keys()
            assert "reconstruction" in decoder_recon.keys()

        else:
            for lev in recon_layers:
                if lev != -1:
                    assert f"embedding_layer_{lev}" in encoder_embed.keys()
                    assert f"reconstruction_layer_{lev}" in decoder_recon.keys()

                else:
                    assert "embedding" in encoder_embed.keys()
                    assert "log_covariance" in encoder_embed.keys()
                    assert "reconstruction" in decoder_recon.keys()

        if recon_layers is None:
            assert encoder_embed["embedding"].shape[1] == ae_celeba_config.latent_dim
            assert (
                encoder_embed["log_covariance"].shape[1] == ae_celeba_config.latent_dim
            )
            assert (
                decoder_recon[f"reconstruction"].shape[1:] == ae_celeba_config.input_dim
            )

        else:
            if 1 in recon_layers:
                assert encoder_embed[f"embedding_layer_1"].shape[1] == 64
                assert decoder_recon[f"reconstruction_layer_1"].shape[1] == 128

            if 2 in recon_layers:
                assert encoder_embed[f"embedding_layer_2"].shape[1] == 128
                assert decoder_recon[f"reconstruction_layer_2"].shape[1] == 128

            if 3 in recon_layers:
                assert encoder_embed[f"embedding_layer_3"].shape[1] == 128
                assert decoder_recon[f"reconstruction_layer_3"].shape[1] == 128

            if 4 in recon_layers:
                assert encoder_embed[f"embedding_layer_4"].shape[1] == 128
                assert decoder_recon[f"reconstruction_layer_4"].shape[1] == 128

            if 5 in recon_layers:
                assert encoder_embed[f"embedding_layer_5"].shape[1] == 128
                assert decoder_recon[f"reconstruction_layer_5"].shape[1] == 64

            if 6 in recon_layers:
                assert (
                    decoder_recon[f"reconstruction_layer_6"].shape[1:]
                    == ae_celeba_config.input_dim
                )

            if -1 in recon_layers:
                assert (
                    encoder_embed["embedding"].shape[1] == ae_celeba_config.latent_dim
                )
                assert (
                    encoder_embed["log_covariance"].shape[1]
                    == ae_celeba_config.latent_dim
                )
                assert (
                    decoder_recon[f"reconstruction"].shape[1:]
                    == ae_celeba_config.input_dim
                )

    def test_svae_encoding_decoding(
        self, ae_celeba_config, celeba_like_data, recon_layers
    ):
        encoder = Encoder_ResNet_SVAE_CELEBA(ae_celeba_config).to(device)
        decoder = Decoder_ResNet_AE_CELEBA(ae_celeba_config).to(device)

        output = encoder(celeba_like_data)
        embedding, log_concentration = output.embedding, output.log_concentration

        assert embedding.shape == (
            celeba_like_data.shape[0],
            ae_celeba_config.latent_dim,
        )
        assert log_concentration.shape == (celeba_like_data.shape[0], 1)

        reconstruction = decoder(embedding).reconstruction

        assert reconstruction.shape == celeba_like_data.shape

        encoder_embed = encoder(celeba_like_data, output_layer_levels=recon_layers)
        decoder_recon = decoder(embedding, output_layer_levels=recon_layers)

        if recon_layers is None:
            assert "embedding" in encoder_embed.keys()
            assert "log_concentration" in encoder_embed.keys()
            assert "reconstruction" in decoder_recon.keys()

        else:
            for lev in recon_layers:
                if lev != -1:
                    assert f"embedding_layer_{lev}" in encoder_embed.keys()
                    assert f"reconstruction_layer_{lev}" in decoder_recon.keys()

                else:
                    assert "embedding" in encoder_embed.keys()
                    assert "log_concentration" in encoder_embed.keys()
                    assert "reconstruction" in decoder_recon.keys()

        if recon_layers is None:
            assert encoder_embed["embedding"].shape[1] == ae_celeba_config.latent_dim
            assert encoder_embed["log_concentration"].shape[1] == 1
            assert (
                decoder_recon[f"reconstruction"].shape[1:] == ae_celeba_config.input_dim
            )

        else:
            if 1 in recon_layers:
                assert encoder_embed[f"embedding_layer_1"].shape[1] == 64
                assert decoder_recon[f"reconstruction_layer_1"].shape[1] == 128

            if 2 in recon_layers:
                assert encoder_embed[f"embedding_layer_2"].shape[1] == 128
                assert decoder_recon[f"reconstruction_layer_2"].shape[1] == 128

            if 3 in recon_layers:
                assert encoder_embed[f"embedding_layer_3"].shape[1] == 128
                assert decoder_recon[f"reconstruction_layer_3"].shape[1] == 128

            if 4 in recon_layers:
                assert encoder_embed[f"embedding_layer_4"].shape[1] == 128
                assert decoder_recon[f"reconstruction_layer_4"].shape[1] == 128

            if 5 in recon_layers:
                assert encoder_embed[f"embedding_layer_5"].shape[1] == 128
                assert decoder_recon[f"reconstruction_layer_5"].shape[1] == 64

            if 6 in recon_layers:
                assert (
                    decoder_recon[f"reconstruction_layer_6"].shape[1:]
                    == ae_celeba_config.input_dim
                )

            if -1 in recon_layers:
                assert (
                    encoder_embed["embedding"].shape[1] == ae_celeba_config.latent_dim
                )
                assert encoder_embed["log_concentration"].shape[1] == 1
                assert (
                    decoder_recon[f"reconstruction"].shape[1:]
                    == ae_celeba_config.input_dim
                )

    def test_vqvae_encoding_decoding(
        self, ae_celeba_config, celeba_like_data, recon_layers
    ):
        encoder = Encoder_ResNet_VQVAE_CELEBA(ae_celeba_config).to(device)
        decoder = Decoder_ResNet_VQVAE_CELEBA(ae_celeba_config).to(device)

        embedding = encoder(celeba_like_data).embedding

        assert embedding.shape[:2] == (
            celeba_like_data.shape[0],
            ae_celeba_config.latent_dim,
        )

        reconstruction = decoder(embedding).reconstruction

        assert reconstruction.shape == celeba_like_data.shape

        encoder_embed = encoder(celeba_like_data, output_layer_levels=recon_layers)
        decoder_recon = decoder(embedding, output_layer_levels=recon_layers)

        if recon_layers is None:
            assert "embedding" in encoder_embed.keys()
            assert "reconstruction" in decoder_recon.keys()

        else:
            for lev in recon_layers:
                if lev != -1:
                    assert f"embedding_layer_{lev}" in encoder_embed.keys()
                    assert f"reconstruction_layer_{lev}" in decoder_recon.keys()

                else:
                    assert "embedding" in encoder_embed.keys()
                    assert "reconstruction" in decoder_recon.keys()

        if recon_layers is None:
            assert encoder_embed["embedding"].shape[1] == ae_celeba_config.latent_dim
            assert (
                decoder_recon[f"reconstruction"].shape[1:] == ae_celeba_config.input_dim
            )

        else:
            if 1 in recon_layers:
                assert encoder_embed[f"embedding_layer_1"].shape[1] == 64
                assert decoder_recon[f"reconstruction_layer_1"].shape[1] == 128

            if 2 in recon_layers:
                assert encoder_embed[f"embedding_layer_2"].shape[1] == 128
                assert decoder_recon[f"reconstruction_layer_2"].shape[1] == 128

            if 3 in recon_layers:
                assert encoder_embed[f"embedding_layer_3"].shape[1] == 128
                assert decoder_recon[f"reconstruction_layer_3"].shape[1] == 128

            if 4 in recon_layers:
                assert encoder_embed[f"embedding_layer_4"].shape[1] == 128
                assert decoder_recon[f"reconstruction_layer_4"].shape[1] == 128

            if 5 in recon_layers:
                assert encoder_embed[f"embedding_layer_5"].shape[1] == 128
                assert decoder_recon[f"reconstruction_layer_5"].shape[1] == 64

            if 6 in recon_layers:
                assert (
                    decoder_recon[f"reconstruction_layer_6"].shape[1:]
                    == ae_celeba_config.input_dim
                )

            if -1 in recon_layers:
                assert (
                    encoder_embed["embedding"].shape[1] == ae_celeba_config.latent_dim
                )
                assert (
                    decoder_recon[f"reconstruction"].shape[1:]
                    == ae_celeba_config.input_dim
                )
