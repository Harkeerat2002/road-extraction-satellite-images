# CS-433 Machine Learning - Beyond Pixel Classification: Enforcing Topological Connectivity in Aerial Road Extraction in Limited Data Regimes

This is the code for the Road Segmentation Project (Project 2) in CS 433 Machine learning.

The AICrowd platform was used for testing (https://www.aicrowd.com/challenges/epfl-ml-road-segmentation)

## Data
First make sure you have the data inside data folder in this structure
```
data
├── test_set_images
│   ├── test_1
│   │   └── test_1.png
│   ├── ...
│   │   └── ...
│   ├── test_50
│   │   └── test_50.png
└── training
    ├── groundtruth
    │   ├── satImage_001.png
    │   ├── ...
    │   └── satImage_100.png
    └── images
        ├── satImage_001.png
        ├── ...
        └── satImage_100.png
```

## Installation instructions

First create a virtual environment with python 3.12.12. Then run:

```bash
bash install.sh
```

## Structure of the repo

Each model has a separate config in configs.py. Model definitions are in `models.py`

The training and inference script uses a configuration ID to select the model architecture and hyperparameters. Use the table below to select the correct `config_id`.:

| Config ID | Model Name | Description |
| :---: | :--- | :--- |
| **0** | **LinkNet** | Standard LinkNet architecture with data augmentation. |
| **1** | **D-LinkNet** | LinkNet with a central Dilation Block (D-Block). |
| **2** | **D-LinkNetMTL** | D-LinkNet with Multi-Task Learning (Orientation). |
| **3** | **D-LinkNetRes34MTL** | D-LinkNetMTL with a Pretrained ResNet34 Encoder. |
| **4** | **SpinD-LinkNetRes34MTL** | Pretrained ResNet34 Encoder D-LinkNetMTL model with SPIN attention module. |
| **5** | **LinkNet (without Augmentation)** | LinkNet trained without data augmentation. |
| **6** | **U-Net** | Standard U-Net architecture. |
| **7** | **Plain CNN** | Simple Convolutional Neural Network baseline. |
| **8** | **Logistic Regression** | Linear baseline. |

## Inference

To run inference, first download the trained model weights from the provided Google Drive link given below and put the downlaoded in the `checkpoints` directory.

```
checkpoints
├── final_model.pth
```

Then execute the `inference.py` script with the config ID of the model.
Example:

```bash
python inference.py --config <config_id>
```

Please replace `<config_id>` with the config ID of the model and put the trained model weights in the `checkpoints` directory.

## Train

Execute the `train.py` script with `crossval` 0 and config ID of the model.

```bash
python train.py --crossval 0 --config <config_id>
```

Please replace `<config_id>` with the config ID of the model.

## 5-Fold Cross Validation

Execute the `train.py` script with `crossval` 1 and config ID of the model.

```bash
python train.py --crossval 1 --config <config_id>
```

Please replace `<config_id>` with the config ID of the model.

## Replicating the Best Model Results

You can download the trained model from the provided Google Drive link's D-LinkNetRes34MTL folder and put the trained model weights in the `checkpoints` directory Or You can train the model D-LinkNetRes34MTL.

```bash
python train.py --crossval 0 --config 3
```

Then, run inference.py on the trained model:

```bash
python inference.py --config 3
```

## Google Drive Download Link

The trained models and their test predictions (submmitted on AICrowd) are present in their respective folders in the link:
https://drive.google.com/drive/folders/1xZiL0sIXOmyJODcUqneu0C-oeKCmTvWQ?usp=sharing

## References

This project utilizes the **ResNet-34** architecture with pre-trained weights provided by Torchvision.

**Model Architecture:**
> K. He, X. Zhang, S. Ren and J. Sun, "Deep Residual Learning for Image Recognition," in CVPR, 2016, pp. 770-778.

**Implementation & Weights:**
> TorchVision maintainers and contributors. (2016). TorchVision: PyTorch's Computer Vision library [Computer software]. https://github.com/pytorch/vision
