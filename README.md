# Traffic4Cast2021-SwinUNet3D (AI4EX Team)

## Table of Content
* [General Info](#general-info)
* [Requirements](#requirements)
* [Installation](#installation)
* [Usage](#usage)
* [Inference](#inference)

## General Info
This resipository contains our code submitted to Traffic4cast2021 competition (https://www.iarai.ac.at/traffic4cast/2021-competition/challenge/#challenge)
This work is made available under the attached license

## Requirements
This resipository depends on the following packages availability
- Pytorch Lightning
- timm
- torch_optimizer
- pytorch_model_summary
- einops

## Installation:
```
unzip folder.zip
cd folder
conda create --name swinencoder_env python=3.6
conda activate swinencoder_env
conda install pytorch=1.9.0 cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
```

## Usage
- a.1) train from scratch
    ```
    python main.py --gpus 0 --use_all_region
    ```
- a.2) fine tune a model from a checkpoint
    ```
    python main.py --gpu_id 1 --use_all_region --mode train --name ALL_real_swinencoder3d_688080 --time-code 20210630T224355 --initial-epoch 58```
    
- b.1) evaluate an untrained model (with random weights)
    ```
    python main.py --gpus 0 --use_all_region --mode test
    ```
- b.2) evaluate a trained model from a checkpoint (submitted inference)
    ```
    python main.py --gpu_id 1 --use_all_region --mode test --name ALL_real_swinencoder3d_688080 --time-code 20210630T224355 --initial-epoch 58
    ```
 
## Inference
To generate predictions using our trained model
```
R=R1
INPUT_PATH=../data
WEIGHTS=logs/ALL_real_swinencoder3d_688080
OUT_PATH=.
python inference.py -d $INPUT_PATH -r $R -w $WEIGHTS -o $OUT_PATH -g 1
```
