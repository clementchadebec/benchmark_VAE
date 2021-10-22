#!/bin/bash

#python training.py --dataset mnist --model_name ae --model_config 'mnist/configs/ae_config.json' --training_config 'mnist/configs/base_training_config.json'
#python training.py --dataset mnist --model_name vae --model_config 'configs/mnist/vae_config.json' --training_config 'configs/mnist/base_training_config.json'
#python training.py --dataset mnist --model_name wae --model_config 'configs/mnist/wae_config.json' --training_config 'configs/mnist/base_training_config.json'
#python training.py --dataset mnist --model_name vamp --model_config 'configs/mnist/vamp_config.json' --training_config 'configs/mnist/base_training_config.json'
#python training.py --dataset mnist --model_name rae_l2 --model_config 'configs/mnist/rae_l2_config.json' --training_config 'configs/mnist/base_training_config.json'
#python training.py --dataset mnist --model_name rae_gp --model_config 'configs/mnist/rae_gp_config.json' --training_config 'configs/mnist/base_training_config.json'
python training.py --dataset mnist --model_name hvae --model_config 'configs/mnist/hvae_config.json' --training_config 'configs/mnist/base_training_config.json'
python training.py --dataset mnist --model_name rhvae --model_config 'configs/mnist/rhvae_config.json' --training_config 'configs/mnist/base_training_config.json'
