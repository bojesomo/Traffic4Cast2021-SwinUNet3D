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
- a.1)train from scratch (together with inference predictions)
    ```
    python Traffic4Cast2021/main1.py --nodes 1 --gpus 4 --precision 16 --batch-size 5 --epochs 100 --mlp_ratio 1 --stages 4 --patch_size 4 --dropout 0.0 --start_filters 192 --sampling-step 1 --decode_depth 1 --use_neck --lr 1e-4 --optimizer lamb --merge_type both --mix_features --city_category TEMPORAL --memory_efficient
    ```
- a.2) fine tune a model from a checkpoint
    ```
    python main.py --gpus 1 --city_category TEMPORAL --mode train --name TEMPORAL_real_swinunet3d_141848694 --time-code 20210913T135845 --initial-epoch 36```

- b) evaluate a trained model from a checkpoint (submitted inference)
    ```
    python main.py --gpus 1 --city_category TEMPORAL --mode test --name TEMPORAL_real_swinunet3d_141848694 --time-code 20210913T135845 --initial-epoch 36
    ```
 
## Inference
- a) To generate predictions using our trained model
```
python main.py --gpus 1 --city_category TEMPORAL --mode test --name TEMPORAL_real_swinunet3d_141848694 --time-code 20210913T135845 --initial-epoch 36
```

- b) To create submission in form of a zipped file from files generater in (a)
```
python create_submission.py --name TEMPORAL_real_swinunet3d_141848694 --time-code 20210913T135845 --epoch 36
```
