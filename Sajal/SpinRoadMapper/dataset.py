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
        if fold is not None:
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

    def getRoadData(self, index):

        image_dict = self.files[self.split][index]
        # read each image in list
        if os.path.isfile(image_dict["img"]):
            image = cv2.imread(image_dict["img"]).astype(np.float32)
        else:
            print("ERROR: couldn't find image -> ", image_dict["img"])

        if os.path.isfile(image_dict["lbl"]):
            gt = cv2.imread(image_dict["lbl"], 0).astype(np.float32)
            gt = binarize(gt, 128).astype(np.float32)

        else:
            print("ERROR: couldn't find image -> ", image_dict["lbl"])

        if self.split == "train":
            image, gt = self.random_crop(image, gt, self.crop_size)
        else:
            image = cv2.resize(
                image,
                (self.crop_size[0], self.crop_size[1]),
                interpolation=cv2.INTER_LINEAR,
            )
            gt = cv2.resize(
                gt,
                (self.crop_size[0], self.crop_size[1]),
                interpolation=cv2.INTER_LINEAR,
            )

        if self.split == "train" and index == len(self.files[self.split]) - 1:
            np.random.shuffle(self.files[self.split])

        h, w, c = image.shape
        if self.augmentation == 1:
            flip = np.random.choice(2) * 2 - 1
            image = np.ascontiguousarray(image[:, ::flip, :])
            gt = np.ascontiguousarray(gt[:, ::flip])
            rotation = np.random.randint(4) * 90
            M = cv2.getRotationMatrix2D((w / 2, h / 2), rotation, 1)
            image = cv2.warpAffine(image, M, (w, h))
            gt = cv2.warpAffine(gt, M, (w, h))

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

    def random_crop(self, image, gt, size):

        w, h, _ = image.shape
        crop_h, crop_w = size

        start_x = np.random.randint(0, w - crop_w)
        start_y = np.random.randint(0, h - crop_h)

        image = image[start_x : start_x + crop_w, start_y : start_y + crop_h, :]
        gt = gt[start_x : start_x + crop_w, start_y : start_y + crop_h]

        return image, gt


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
                gt_ = cv2.resize(
                    gt,
                    (int(math.ceil(h / (val * 1.0))), int(math.ceil(w / (val * 1.0)))),
                    interpolation=cv2.INTER_NEAREST,
                )
            else:
                gt_ = gt

            gt_orig = np.copy(gt_)
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