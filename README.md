# Look-Ahead active learning with Neural Tangent Kernels

This repository contains an implementation of the proposed method for the paper ["Making Look-Ahead Active Learning Strategies Feasible with Neural Tangent Kernels"](https://arxiv.org/abs/2206.12569), to appear at NeurIPS 2022 by Mohamad Amin Mohamadi*, Wonhoe Bae* and Danica J. Sutherland.

## Configs
Configs are written in the form of yaml. Please refer to the `configs/cifar_resnet18.yml` for an exmaple config and for the details about how to structure configs.

## Datasets
Datasets are supposed to be located in data root which can be modified in the data section of a config file.

## Train
To run the proposed method with the desired configuraiton, please refer to the command below. 
```bash
python run.py --config_path configs/cifar_resnet18.yml --save_dir /tmp
```
