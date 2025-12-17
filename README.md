# Road Segmentation From Satellite Images

This is the code for the course project in CS 433 (Machine learning).

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

First create a virtual environment with python 3.12.12.

```bash
bash install.sh
```

## Structure of the repo

Each model has a separate config in configs.py. Model definitions are in `models.py`

This repo contains these models (and their config number in configs.py):

*   Logistic Regression
*   Plain CNN
*   LinkNet (config number 0)
*   D-LinkNet (config number 1)
*   D-LinkNetMTL (config number 2)
*   D-LinkNetRes34MTL (config number 3)
*   SpinD-LinkNetRes34MTL (config number 4)
*   LinkNet (without Augmentation) (config number 5)
*   U-Net (config number 6)


## Inference

To run inference, first download the trained model weights from the provided Google Drive link given below and put the downlaoded in the `checkpoints` directory.

```
checkpoints
├── final_model.pth
```

Then execute the `inference.py` script with the config number of the model.
Example:

```bash
python inference.py --config <config_number>
```

Please replace `<config_number>` with the config number of the model and put the trained model weights in the `checkpoints` directory.

## Train

Execute the `train.py` script with the config number of the model.

```bash
python train.py --crossval 0 --config <config_number>
```

Please replace `<config_number>` with the config number of the model.

## 5-Fold Cross Validation

Execute the `train.py` script with the config number of the model.

```bash
python train.py --crossval 1 --config <config_number>
```

Please replace `<config_number>` with the config number of the model.

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
