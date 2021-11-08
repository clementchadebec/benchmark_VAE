import pytest
import torch

from pythae.models import AEConfig, VAEConfig
from pythae.models.nn.benchmarks.mnist import *
from pythae.models.nn.benchmarks.celeba import *
from pythae.models.nn.benchmarks.cifar import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'



class Test_MNIST_Benchmark:
    @pytest.fixture(
    params=[
        AEConfig(input_dim=(1, 28, 28), latent_dim=10),
        AEConfig(input_dim=(1, 28, 28), latent_dim=5),
    ]
    )
    def ae_mnist_config(self, request):
        return request.param

    @pytest.fixture()
    def mnist_like_data(self):
        return torch.rand(3, 1, 28, 28).to(device)

    def test_ae_encoding_decoding(self, ae_mnist_config, mnist_like_data):
        encoder = Encoder_AE_MNIST(ae_mnist_config)
        decoder = Decoder_AE_MNIST(ae_mnist_config)

        embedding = encoder(mnist_like_data).embedding
    
        assert embedding.shape == (mnist_like_data.shape[0], ae_mnist_config.latent_dim)

        reconstruction = decoder(embedding).reconstruction

        assert reconstruction.shape == mnist_like_data.shape

    
    @pytest.fixture(
    params=[
        VAEConfig(input_dim=(1, 28, 28), latent_dim=10),
        VAEConfig(input_dim=(1, 28, 28), latent_dim=5),
    ]
    )
    def vae_mnist_config(self, request):
        return request.param

    def test_vae_encoding_decoding(self, vae_mnist_config, mnist_like_data):
        encoder = Encoder_VAE_MNIST(vae_mnist_config)
        decoder = Decoder_AE_MNIST(vae_mnist_config)

        output = encoder(mnist_like_data)
        embedding, log_covariance = output.embedding, output.log_covariance
    
        assert embedding.shape == (mnist_like_data.shape[0], vae_mnist_config.latent_dim)
        assert log_covariance.shape == (mnist_like_data.shape[0], vae_mnist_config.latent_dim)

        reconstruction = decoder(embedding).reconstruction

        assert reconstruction.shape == mnist_like_data.shape

class Test_CIFAR_Benchmark:
    @pytest.fixture(
    params=[
        AEConfig(input_dim=(1, 28, 28), latent_dim=10),
        AEConfig(input_dim=(1, 28, 28), latent_dim=5),
    ]
    )
    def ae_cifar_config(self, request):
        return request.param

    @pytest.fixture()
    def cifar_like_data(self):
        return torch.rand(3, 3, 32, 32).to(device)

    def test_ae_encoding_decoding(self, ae_cifar_config, cifar_like_data):
        encoder = Encoder_AE_CIFAR(ae_cifar_config)
        decoder = Decoder_AE_CIFAR(ae_cifar_config)

        embedding = encoder(cifar_like_data).embedding
    
        assert embedding.shape == (cifar_like_data.shape[0], ae_cifar_config.latent_dim)

        reconstruction = decoder(embedding).reconstruction

        assert reconstruction.shape == cifar_like_data.shape

    @pytest.fixture(
    params=[
        VAEConfig(input_dim=(3, 32, 32), latent_dim=10),
        VAEConfig(input_dim=(3, 32, 32), latent_dim=5),
    ]
    )
    def vae_cifar_config(self, request):
        return request.param

    def test_vae_encoding_decoding(self, vae_cifar_config, cifar_like_data):
        encoder = Encoder_VAE_CIFAR(vae_cifar_config)
        decoder = Decoder_AE_CIFAR(vae_cifar_config)

        output = encoder(cifar_like_data)
        embedding, log_covariance = output.embedding, output.log_covariance
    
        assert embedding.shape == (cifar_like_data.shape[0], vae_cifar_config.latent_dim)
        assert log_covariance.shape == (cifar_like_data.shape[0], vae_cifar_config.latent_dim)

        reconstruction = decoder(embedding).reconstruction

        assert reconstruction.shape == cifar_like_data.shape


class Test_CELEBA_Benchmark:
    @pytest.fixture(
    params=[
        AEConfig(input_dim=(3, 64, 64), latent_dim=10),
        AEConfig(input_dim=(3, 64, 64), latent_dim=5),
    ]
    )
    def ae_celeba_config(self, request):
        return request.param

    @pytest.fixture()
    def celeba_like_data(self):
        return torch.rand(3, 3, 64, 64).to(device)

    def test_ae_encoding_decoding(self, ae_celeba_config, celeba_like_data):
        encoder = Encoder_AE_CELEBA(ae_celeba_config)
        decoder = Decoder_AE_CELEBA(ae_celeba_config)

        embedding = encoder(celeba_like_data).embedding
    
        assert embedding.shape == (celeba_like_data.shape[0], ae_celeba_config.latent_dim)

        reconstruction = decoder(embedding).reconstruction

        assert reconstruction.shape == celeba_like_data.shape


    @pytest.fixture(
    params=[
        VAEConfig(input_dim=(3, 64, 64), latent_dim=10),
        VAEConfig(input_dim=(3, 64, 64), latent_dim=5),
    ]
    )
    def vae_celeba_config(self, request):
        return request.param

    def test_vae_encoding_decoding(self, vae_celeba_config, celeba_like_data):
        encoder = Encoder_VAE_CELEBA(vae_celeba_config)
        decoder = Decoder_AE_CELEBA(vae_celeba_config)

        output = encoder(celeba_like_data)
        embedding, log_covariance = output.embedding, output.log_covariance
    
        assert embedding.shape == (celeba_like_data.shape[0], vae_celeba_config.latent_dim)
        assert log_covariance.shape == (celeba_like_data.shape[0], vae_celeba_config.latent_dim)

        reconstruction = decoder(embedding).reconstruction

        assert reconstruction.shape == celeba_like_data.shape
