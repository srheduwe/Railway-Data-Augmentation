# Railway-Data-Augmentation
Welcome to the accompagning GitHub repository to our paper "Data Augmentation for Railway Applications using Diffusion Models".

## Install dependencies

Getting started: To create an Anaconda environment with the necessary dependencies follow these steps:

```
conda create -n Aug python=3.9.19
conda activate Aug
pip install -r requirements.txt
```

## Download UAV-RSOD dataset

Download the open-source UAV-RSOD dataset from Rampriya et al. from [Zenodo](https://zenodo.org/records/12606374) and add to ```data/nature_data/``` folder.

## Classifier fine-tuning

Launch fine-tuning of the image classifier architectures by executing the following commands:

```bash
# Args:
#   model:  rn50, rrn, vit
#   data:   original, baseline, synthetic, synthetic+

python src/train_classifiers.py model="rn50" data="baseline"
```

## Project Organization


    ├── README.md                   <- The top-level README for developers using this project.
    │
    ├── config                      <- Hydra condiguration files for model fine-tuning and label-to-index mapping
    │   ├── data                    <- Dataset configuration
    │   │   └── original.yaml       <- Original dataset
    │   │   └── baseline.yaml       <- Baseline dataset
    │   │   └── synthetic.yaml      <- Synthetic dataset
    │   │   └── synthetic+.yaml     <- Synthetic+ dataset
    │   ├── model                   <- Model configuration
    │   │   └── rn50.yaml           <- ResNet-50
    │   │   └── rrn.yaml            <- RobustRailNet
    │   │   └── vit.yaml            <- Vision Transformer
    │   ├── training                <- Training configuration
    │   │   └── train.yaml          <- Training parameters
    │   └── label2idx.json          <- Label-to-index mapping of dataset classes
    │
    ├── Dataset-examples            <- Folder containing 10 example images of each class.
    │   └── Branch                  <- Images of the class Branch
    │   └── Barrel                  <- Images of the class Barrel
    │   └── etc..                   <- Other classes
    │
    ├── requirements.txt            <- The requirements file for reproducing the analysis environment.
    │
    └──  src                        <- Source code for use in this project.
         └── caption_baseline.py    <- Script for generating the rule-based prompts
         └── caption_variation.py   <- Script for generating and varying prompts using LLMs
         └── flux_submit.py         <- Generates images with Flux 2 by sending them on an HPC in batches using submitit
         └── HPO.py                 <- Hyperparameter Optimisation with SMAC3
         └── model_RN50.py          <- ResNet-50 architecture
         └── model_RRN.py           <- RobustRailNet architecture
         └── model_ViT.py           <- Vision Transformer architecture
         └── train_classifiers.py   <- Fine-tuning of image classifier architectures
         └── train_flux.py          <- Fine-tuning Flux
         └── utils.py               <- Utility module for model fine-tuning

