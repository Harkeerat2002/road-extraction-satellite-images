config = {
    "seed": 7,
    "task1_classes": 2,
    "task2_classes": 37,
    "task1_weight": 1,
    "task2_weight": 1,
    "train_batch_size": 4,
    "val_batch_size": 4,
    "refinement":3,
    "model_name": "StackHourglassNetMTL",
    "checkpoint_path": "/content/checkpoints/final_model.pth",
    "test_dir": "/content/drive/MyDrive/CS433/test_set_images/",
    "output_dir": "/content/results/",
    "nfolds": 10,
    "train_dataset": {
        "deepglobe":{
            "dir": "/content/drive/MyDrive/CS433/training/",
            "crop_size": 256
        },
        "crop_size": 256,
        "augmentation": True,
        "mean" : "[70.95016901, 71.16398124, 71.30953645]",
        "std" : "[ 34.00087859, 35.18201658, 36.40463264]",
        "normalize_type": "Mean",
        "thresh": 0.76,
        "angle_theta": 10,
        "angle_bin": 10
    },
    "val_dataset": {
        "deepglobe":{
            "dir": "/content/drive/MyDrive/CS433/training/",
            "crop_size": 400
        },
        "crop_size": 400,
        "augmentation": False,
        "mean" : "[70.95016901, 71.16398124, 71.30953645]",
        "std" : "[ 34.00087859, 35.18201658, 36.40463264]",
        "normalize_type": "Mean",
        "thresh": 0.76,
        "angle_theta": 10,
        "angle_bin": 10
    },
    "optimizer": {
        "lr": 0.01,
        "d_lr": 0.0001,
        "lr_step": 0.1,
        "lr_drop_epoch": "[60,90,110]"
    },
    "trainer": {
        "total_epochs": 120,
        "logging_dir": "/content/logs/",
        "iter_size": 1,
        "test_freq": 1
    }
}