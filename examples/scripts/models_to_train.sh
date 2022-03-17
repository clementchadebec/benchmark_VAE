#!/bin/bash

#python training.py --dataset mnist --model_name vamp --model_config 'configs/mnist/vamp_config.json' --training_config 'configs/mnist/base_training_config.json'
#python training.py --dataset mnist --model_name info_vae --model_config 'configs/mnist/info_vae_config.json' --training_config 'configs/mnist/base_training_config.json'

#python training.py --dataset mnist --model_name ae --model_config 'configs/mnist/ae_config.json' --training_config 'configs/mnist/base_training_config.json'
#python training.py --dataset mnist --model_name vae --model_config 'configs/mnist/vae_config.json' --training_config 'configs/mnist/base_training_config.json'
#python training.py --dataset mnist --model_name wae --model_config 'configs/mnist/wae_config.json' --training_config 'configs/mnist/base_training_config.json'
#python training.py --dataset mnist --model_name rae_l2 --model_config 'configs/mnist/rae_l2_config.json' --training_config 'configs/mnist/base_training_config.json'
#python training.py --dataset mnist --model_name rae_gp --model_config 'configs/mnist/rae_gp_config.json' --training_config 'configs/mnist/base_training_config.json'
#python training.py --dataset mnist --model_name iwae --model_config 'configs/mnist/iwae_config.json' --training_config 'configs/mnist/base_training_config.json'
#python training.py --dataset mnist --model_name hvae --model_config 'configs/mnist/hvae_config.json' --training_config 'configs/mnist/base_training_config.json'
#python training.py --dataset mnist --model_name rhvae --model_config 'configs/mnist/rhvae_config.json' --training_config 'configs/mnist/base_training_config.json'


#python training.py --dataset mnist --model_name beta_vae --model_config configs/mnist/beta_vae_config.json --training_config configs/mnist/base_training_config.json
#python training.py --dataset mnist --model_name aae --model_config configs/mnist/aae_config.json --training_config configs/mnist/base_training_config.json
#python training.py --dataset celeba --model_name aae --model_config configs/celeba/aae_config.json --training_config configs/celeba/base_training_config.json

#python training.py --dataset celeba --model_name ae --model_config 'configs/celeba/ae_config.json' --training_config 'configs/celeba/base_training_config.json'
#python training.py --dataset celeba --model_name vae --model_config 'configs/celeba/vae_config.json' --training_config 'configs/celeba/base_training_config.json'
#python training.py --dataset celeba --model_name wae --model_config 'configs/celeba/wae_config.json' --training_config 'configs/celeba/base_training_config.json'
##python training.py --dataset celeba --model_name vamp --model_config 'configs/celeba/vamp_config.json' --training_config 'configs/celeba/base_training_config.json'
#python training.py --dataset celeba --model_name rae_l2 --model_config 'configs/celeba/rae_l2_config.json' --training_config 'configs/celeba/base_training_config.json'
#python training.py --dataset celeba --model_name rae_gp --model_config 'configs/celeba/rae_gp_config.json' --training_config 'configs/celeba/base_training_config.json'
#python training.py --dataset celeba --model_name hvae --model_config 'configs/celeba/hvae_config.json' --training_config 'configs/celeba/base_training_config.json'
#python training.py --dataset celeba --model_name rhvae --model_config 'configs/celeba/rhvae_config.json' --training_config 'configs/celeba/base_training_config.json'
#python training.py --dataset celeba --model_name vqvae --model_config 'configs/celeba/vqvae_config.json' --training_config 'configs/celeba/base_training_config.json'

python training.py --dataset mnist --model_name beta_tc_vae --model_config configs/mnist/beta_tc_vae_config.json --training_config configs/mnist/base_training_config.json
python training.py --dataset celeba --model_name beta_tc_vae --model_config configs/celeba/beta_tc_vae_config.json --training_config configs/celeba/base_training_config.json