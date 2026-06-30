# Railway-Data-Augmentation
Welcome to the accompagning GitHub repository to our paper "Data Augmentation for Railway Applications using Diffusion Models".

Getting started: To create an Anaconda environment with the necessary dependencies follow these steps:


```
conda create -n Aug python=3.9.19
conda activate Aug
pip install -r requirements.txt
```



Project Organization
------------

    ├── README.md                   <- The top-level README for developers using this project.
    ├── Dataset-examples            <- Folder containing 10 example images of each class.
    │   └── Branch                  <- Images of the class Branch
    │   └── Barrel                  <- Images of the class Barrel
    │   └── etc..                   <- Other classes
    │
    ├── requirements.txt            <- The requirements file for reproducing the analysis environment.
    │
    └──  src                        <- Source code for use in this project.
         └── HPO.py                 <- Hyperparameter Optimisation with SMAC3
         └── caption_baseline.py    <- Script for generating the rule-based prompts
         └── caption_variation.py   <- Script for generating and varying prompts using LLMs
         └── flux_submit.py         <- Generates images with Flux 2 by sending them on an HPC in batches using submitit
         └── train_flux.py          <- Fine-tuning Flux
