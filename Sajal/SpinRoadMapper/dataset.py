import collections
import math
import os
import random

import cv2
import numpy as np
import torch
from torch.utils import data
from orientation_utils import getVectorMapsAngles, getKeypoints
from util import binarize
import albumentations as A 


class RoadDataset(data.Dataset):
    def __init__(
        self, 
        config, 
        dataset_name, 
        seed, 
        multi_scale_pred, 
        is_train, 
        fold, 
        n_folds
    ):
        # Seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

        self.fold = fold
        self.n_folds = n_folds

        self.split = "train" if is_train else "val"
        self.config = config
        
        # paths
        self.file_path = self.config[dataset_name]["dir"]

        # augmentations
        self.augmentation = self.config["augmentation"]
        self.crop_size = [
            self.config[dataset_name]["crop_size"],
            self.config[dataset_name]["crop_size"],
        ]
        self.multi_scale_pred = multi_scale_pred

        # preprocess
        self.angle_theta = self.config["angle_theta"]
        self.mean_bgr = np.array(eval(self.config["mean"]))
        self.deviation_bgr = np.array(eval(self.config["std"]))
        self.normalize_type = self.config["normalize_type"]

        # to avoid Deadloack  between CV Threads and Pytorch Threads caused in resizing
        cv2.setNumThreads(0)

        # Initialize Albumentations Transforms
        self.transforms = self.define_transforms(seed)

        self.colour_transforms = A.Compose(
            [
                A.ColorJitter(brightness=0.4, contrast = 0.4, hue = 0.2, p = 0.5, ),
            ],
            seed = seed
            )

        self.files = collections.defaultdict(list)
        
        # Load all 100 images
        all_files = []
        for i in range(1, 101):  # 100 images
            all_files.append({
                "img": os.path.join(self.file_path, f"images/satImage_{i:03d}.png"),
                "lbl": os.path.join(self.file_path, f"groundtruth/satImage_{i:03d}.png")
            })

        # Split based on fold
        self.files = collections.defaultdict(list)
        if n_folds != 1:
            fold_size = len(all_files) // n_folds
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < n_folds - 1 else len(all_files)
            self.files["val"] = all_files[val_start:val_end]
            self.files["train"] = all_files[:val_start] + all_files[val_end:]
        else:
            self.files["train"] = all_files
            self.files["val"] = []

    def __len__(self):
        return len(self.files[self.split])
    
    def define_transforms(self, seed):
        """
        Defines the Albumentations transformation pipeline.
        """

        transforms_list = []
        h, w = self.crop_size

        if self.split == "train":
            # 1. Random Crop
            transforms_list.append(A.RandomCrop(height=h, width=w))

            # 2. Augmentations (Flip, Rotate, etc.)
            if self.augmentation == 1:
                transforms_list.extend([
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5, border_mode=cv2.BORDER_REFLECT_101),
                    A.CoarseDropout(num_holes_range = (3,6), hole_height_range = (8,15), hole_width_range = (8,15), p = 0.2),
                    A.ElasticTransform(p=0.2),
                ])
        else:
            # For validation, strictly resize to crop_size (or CenterCrop if preferred)
            transforms_list.append(A.Resize(height=h, width=w))

        return A.Compose(transforms_list, seed=seed)

    def getRoadData(self, index):

        image_dict = self.files[self.split][index]
        # read each image in list
        if os.path.isfile(image_dict["img"]):
            image = cv2.imread(image_dict["img"])
        else:
            print("ERROR: couldn't find image -> ", image_dict["img"])

        if os.path.isfile(image_dict["lbl"]):
            gt = cv2.imread(image_dict["lbl"], 0)
            gt = binarize(gt, 128).astype(np.uint8)

        else:
            print("ERROR: couldn't find image -> ", image_dict["lbl"])

        # Geometrical Augmentation
        augmented = self.transforms(image=image, mask=gt)
        image = augmented['image']
        gt = augmented['mask']

        # Color Augmentation (only for training)
        if self.split == "train" and self.augmentation == 1:
            augmentations = self.colour_transforms(image = image)
            image = augmentations["image"]

        image = image.astype(np.float32)
        gt = gt.astype(np.float32)

        # Normalize and reshape
        image = self.reshape(image)
        image = torch.from_numpy(np.array(image))

        return image, gt

    def getOrientationGT(self, keypoints, height, width):
        vecmap, vecmap_angles = getVectorMapsAngles(
            (height, width), keypoints, theta=self.angle_theta, bin_size=10
        )
        vecmap_angles = torch.from_numpy(vecmap_angles)

        return vecmap_angles

    def reshape(self, image):

        if self.normalize_type == "Std":
            image = (image - self.mean_bgr) / (3 * self.deviation_bgr)
        elif self.normalize_type == "MinMax":
            image = (image - self.min_bgr) / (self.max_bgr - self.min_bgr)
            image = image * 2 - 1
        elif self.normalize_type == "Mean":
            image -= self.mean_bgr
        else:
            image = (image / 255.0) * 2 - 1

        image = image.transpose(2, 0, 1)
        return image


class DeepGlobeDataset(RoadDataset):
    def __init__(self, 
                 config, 
                 seed, 
                 multi_scale_pred, 
                 is_train,
                 fold,
                 n_folds
                 ):
        
        super(DeepGlobeDataset, self).__init__(
            config, 
            "deepglobe", 
            seed, 
            multi_scale_pred, 
            is_train, 
            fold, 
            n_folds
            )

    def __getitem__(self, index):

        image, gt = self.getRoadData(index)
        c, h, w = image.shape

        labels = []
        vecmap_angles = []
        if self.multi_scale_pred:
            smoothness = [1, 2, 4]
            scale = [4, 2, 1]
        else:
            smoothness = [4]
            scale = [1]

        for i, val in enumerate(scale):
            if val != 1:
                # Nearest neighbor is safest for masks to preserve edges.
                gt_ = cv2.resize(
                    gt,
                    (int(math.ceil(h / (val * 1.0))), int(math.ceil(w / (val * 1.0)))),
                    interpolation=cv2.INTER_NEAREST,
                )
            else:
                gt_ = gt

            gt_orig = np.copy(gt_)
            # Normalize GT to 0-1 for the loss function
            gt_orig /= 255.0
            labels.append(gt_orig)

            # Create Orientation Ground Truth
            keypoints = getKeypoints(
                gt_orig, is_gaussian=False, smooth_dist=smoothness[i]
            )
            vecmap_angle = self.getOrientationGT(
                keypoints,
                height=int(math.ceil(h / (val * 1.0))),
                width=int(math.ceil(w / (val * 1.0))),
            )
            vecmap_angles.append(vecmap_angle)

        return image, labels, vecmap_angles