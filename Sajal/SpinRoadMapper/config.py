config = {
    "seed": 7,
    "task1_classes": 2,
    "task2_classes": 37,
    "task1_weight": 1,
    "task2_weight": 1,
    "train_batch_size": 4,
    "val_batch_size": 4,
    "model_name": "StackHourglassNetMTL",
    "checkpoint_path": "/content/checkpoints/final_model.pth",
    "test_dir": "/content/drive/MyDrive/CS433/test_set_images/",
    "output_dir": "/content/results/",
    "nfolds": 5,
    "train_dataset": {
        "deepglobe":{
            "dir": "/content/drive/MyDrive/CS433/training/",
            "crop_size": 256
        },
        "augmentation": True,
        "mean" : "[75.42847031, 84.17389662, 84.91022512]",
        "std" : "[47.70426159, 47.7377937, 49.46134476]",
        "normalize_type": "Mean",
        "angle_theta": 10,
        "angle_bin": 10
    },
    "val_dataset": {
        "deepglobe":{
            "dir": "/content/drive/MyDrive/CS433/training/",
            "crop_size": 256
        },
        "augmentation": False,
        "mean" : "[75.42847031, 84.17389662, 84.91022512]",
        "std" : "[47.70426159, 47.7377937, 49.46134476]",
        "normalize_type": "Mean",
        "angle_theta": 10,
        "angle_bin": 10
    },
    "optimizer": {
        "lr": 0.01,
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