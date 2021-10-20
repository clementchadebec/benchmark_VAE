#!/bin/bash

python training.py --dataset mnist --model_name ae --model_config 'mnist/configs/ae_config.json' --training_config 'mnist/configs/base_training_config.json'
python training.py --dataset mnist --model_name vae --model_config 'mnist/configs/vae_config.json' --training_config 'mnist/configs/base_training_config.json'
python training.py --dataset mnist --model_name wae --model_config 'mnist/configs/wae_config.json' --training_config 'mnist/configs/base_training_config.json'
#python training.py --dataset celeba --model_name vamp --model_config 'mnist/configs/vamp_config.json' --training_config 'mnist/configs/base_training_config.json'
python training.py --dataset mnist --model_name hvae --model_config 'mnist/configs/hvae_config.json' --training_config 'mnist/configs/base_training_config.json'
#python training.py --dataset celeba --model_name rhvae --model_config 'mnist/configs/rhvae_config.json' --training_config 'mnist/configs/base_training_config.json'
