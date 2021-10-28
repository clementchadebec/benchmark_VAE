#!/bin/bash
#SBATCH -C v100-16g
#SBATCH --time=2:00:00
#SBATCH --mem=60G
#SBATCH --cpus-per-task=6
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --chdir=.
#SBATCH --output=./outputs/MNIST_VAMP_job_%j.out
#SBATCH --error=./outputs/MNIST_VAMP_job_%j.err
#SBATCH --job-name=VAMP
#SBATCH --gres=gpu:1
#SBATCH -A bgc@gpu

module purge

module load pytorch-gpu/py3/1.9.0

conda activate vae

python training.py --dataset mnist --model_name vamp --model_config 'configs/mnist/vamp_config.json' --training_config 'configs/mnist/base_training_config.json'


#python training.py --dataset celeba --model_name vamp --model_config 'configs/celeba/vamp_config.json' --training_config 'configs/celeba/base_training_config.json'
#python training.py --dataset cifar10 --model_name vae --model_config 'configs/cifar10/vae_config.json' --training_config 'configs/cifar10/base_training_config.json'
#python training.py --dataset celeba --model_name hvae --model_config 'configs/celeba/hvae_config.json' --training_config 'configs/celeba/base_training_config.json'
#python training.py --dataset cifar10 --model_name rae_gp --model_config 'configs/cifar10/rae_gp_config.json' --training_config 'configs/cifar10/base_training_config.json'
#python training.py --dataset cifar10 --model_name rae_l2 --model_config 'configs/cifar10/rae_l2_config.json' --training_config 'configs/cifar10/base_training_config.json'
#python training.py --dataset cifar10 --model_name wae --model_config 'configs/cifar10/wae_config.json' --training_config 'configs/cifar10/base_training_config.json'
#python training.py --dataset cifar10 --model_name ae --model_config 'configs/cifar10/ae_config.json' --training_config 'configs/cifar10/base_training_config.json'

#python training.py --dataset celeba --model_name vamp --model_config 'configs/celeba/vamp_config.json' --training_config 'configs/celeba/base_training_config.json'
#python training.py --dataset celeba --model_name vae --model_config 'configs/celeba/vae_config.json' --training_config 'configs/celeba/base_training_config.json'
#python training.py --dataset celeba --model_name hvae --model_config 'configs/celeba/hvae_config.json' --training_config 'configs/celeba/base_training_config.json'
#python training.py --dataset celeba --model_name rae_gp --model_config 'configs/celeba/rae_gp_config.json' --training_config 'configs/celeba/base_training_config.json'
#python training.py --dataset celeba --model_name rae_l2 --model_config 'configs/celeba/rae_l2_config.json' --training_config 'configs/celeba/base_training_config.json'
#python training.py --dataset celeba --model_name wae --model_config 'configs/celeba/wae_config.json' --training_config 'configs/celeba/base_training_config.json'
#python training.py --dataset celeba --model_name ae --model_config 'configs/celeba/ae_config.json' --training_config 'configs/celeba/base_training_config.json'
