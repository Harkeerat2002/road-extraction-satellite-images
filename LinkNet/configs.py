from path_manager import CHECKPOINT_DIR, LOGGING_DIR, OUTPUT_DIR, TEST_DIR, TRAIN_DIR


configs = [
    {
    "seed": 7,
    "task1_classes": 2,
    "task2_classes": 37,
    "task1_weight": 1,
    "task2_weight": 1,
    "train_batch_size": 8,
    "val_batch_size": 8,
    "model_name": "LinkNet",
    "checkpoint_path": f"{CHECKPOINT_DIR}/final_model.pth",
    "test_dir": TEST_DIR,
    "output_dir": OUTPUT_DIR,
    "nfolds": 5,
    "classification_threshold": 0.5,
    "backbone_pretrained": False,
    "train_dataset": {
        "deepglobe":{
            "dir": TRAIN_DIR,
            "crop_size": 256,
            "stride": 128,
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
            "dir": TRAIN_DIR,
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
        "weight_decay": 0.0005,
    },
    "trainer": {
        "total_epochs": 60,
        "logging_dir": LOGGING_DIR,
    }
},

{
    "seed": 7,
    "task1_classes": 2,
    "task2_classes": 37,
    "task1_weight": 1,
    "task2_weight": 1,
    "train_batch_size": 8,
    "val_batch_size": 8,
    "model_name": "LinkNet",
    "checkpoint_path": f"{CHECKPOINT_DIR}/final_model.pth",
    "test_dir": TEST_DIR,
    "output_dir": OUTPUT_DIR,
    "nfolds": 5,
    "classification_threshold": 0.5,
    "full_frame_inference": True,
    "backbone_pretrained": False,
    "train_dataset": {
        "deepglobe":{
            "dir": TRAIN_DIR,
            "crop_size": 400,
            "stride": 128,
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
            "dir": TRAIN_DIR,
            "crop_size": 400
        },
        "augmentation": False,
        "mean" : "[75.42847031, 84.17389662, 84.91022512]",
        "std" : "[47.70426159, 47.7377937, 49.46134476]",
        "normalize_type": "Mean",
        "angle_theta": 10,
        "angle_bin": 10
    },
    "optimizer": {
        "lr": 1e-3,
        "weight_decay": 0.0005,
    },
    "trainer": {
        "total_epochs": 200,
        "logging_dir": LOGGING_DIR,
    }
},

{
    "seed": 7,
    "task1_classes": 2,
    "task2_classes": 37,
    "task1_weight": 1,
    "task2_weight": 1,
    "train_batch_size": 8,
    "val_batch_size": 8,
    "model_name": "DLinkNet",
    "checkpoint_path": f"{CHECKPOINT_DIR}/final_model.pth",
    "test_dir": TEST_DIR,
    "output_dir": OUTPUT_DIR,
    "nfolds": 5,
    "classification_threshold": 0.5,
    "backbone_pretrained": False,
    "train_dataset": {
        "deepglobe":{
            "dir": TRAIN_DIR,
            "crop_size": 400,
            "stride": 128,
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
            "dir": TRAIN_DIR,
            "crop_size": 400
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
        "weight_decay": 0.0005,
    },
    "trainer": {
        "total_epochs": 60,
        "logging_dir": LOGGING_DIR,
    }
},

{
    "seed": 7,
    "task1_classes": 2,
    "task2_classes": 37,
    "task1_weight": 1,
    "task2_weight": 1,
    "train_batch_size": 8,
    "val_batch_size": 8,
    "model_name": "DLinkNet",
    "checkpoint_path": f"{CHECKPOINT_DIR}/final_model.pth",
    "test_dir": TEST_DIR,
    "output_dir": OUTPUT_DIR,
    "nfolds": 5,
    "classification_threshold": 0.5,
    "backbone_pretrained": False,
    "full_frame_inference": True,
    "train_dataset": {
        "deepglobe":{
            "dir": TRAIN_DIR,
            "crop_size": 400,
            "stride": 32,
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
            "dir": TRAIN_DIR,
            "crop_size": 400
        },
        "augmentation": False,
        "mean" : "[75.42847031, 84.17389662, 84.91022512]",
        "std" : "[47.70426159, 47.7377937, 49.46134476]",
        "normalize_type": "Mean",
        "angle_theta": 10,
        "angle_bin": 10
    },
    "optimizer": {
        "lr": 5e-4,
        "weight_decay": 0.0005,
    },
    "trainer": {
        "total_epochs": 200,
        "logging_dir": LOGGING_DIR,
    }
},

{
    "seed": 7,
    "task1_classes": 2,
    "task2_classes": 37,
    "task1_weight": 1,
    "task2_weight": 1,
    "train_batch_size": 8,
    "val_batch_size": 8,
    "model_name": "DLinkNetMTL",
    "checkpoint_path": f"{CHECKPOINT_DIR}/final_model.pth",
    "test_dir": TEST_DIR,
    "output_dir": OUTPUT_DIR,
    "nfolds": 5,
    "classification_threshold": 0.5,
    "backbone_pretrained": False,
    "train_dataset": {
        "deepglobe":{
            "dir": TRAIN_DIR,
            "crop_size": 400,
            "stride": 128,
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
            "dir": TRAIN_DIR,
            "crop_size": 400
        },
        "augmentation": False,
        "mean" : "[75.42847031, 84.17389662, 84.91022512]",
        "std" : "[47.70426159, 47.7377937, 49.46134476]",
        "normalize_type": "Mean",
        "angle_theta": 10,
        "angle_bin": 10
    },
    "optimizer": {
        "lr": 1e-3,
        "weight_decay": 0.0005,
    },
    "trainer": {
        "total_epochs": 100,
        "logging_dir": LOGGING_DIR,
    }
},

{
    "seed": 7,
    "task1_classes": 2,
    "task2_classes": 37,
    "task1_weight": 1,
    "task2_weight": 1,
    "train_batch_size": 8,
    "val_batch_size": 8,
    "model_name": "DLinkNetMTL",
    "checkpoint_path": f"{CHECKPOINT_DIR}/final_model.pth",
    "test_dir": TEST_DIR,
    "output_dir": OUTPUT_DIR,
    "nfolds": 5,
    "classification_threshold": 0.5,
    "full_frame_inference": True,
    "backbone_pretrained": False,
    "train_dataset": {
        "deepglobe":{
            "dir": TRAIN_DIR,
            "crop_size": 400,
            "stride": 32,
        },
        "augmentation": True,
        "mean" : "[75.42847031, 84.17389662, 84.91022512]",
        "std" : "[47.70426159, 47.7377937, 49.46134476]",
        "normalize_type": "Std",
        "angle_theta": 10,
        "angle_bin": 10
    },
    "val_dataset": {
        "deepglobe":{
            "dir": TRAIN_DIR,
            "crop_size": 400
        },
        "augmentation": False,
        "mean" : "[75.42847031, 84.17389662, 84.91022512]",
        "std" : "[47.70426159, 47.7377937, 49.46134476]",
        "normalize_type": "Std",
        "angle_theta": 10,
        "angle_bin": 10
    },
    "optimizer": {
        "lr": 5e-4,
        "weight_decay": 0.0005,
    },
    "trainer": {
        "total_epochs": 200,
        "logging_dir": LOGGING_DIR,
    }
},

{
    "seed": 7,
    "task1_classes": 2,
    "task2_classes": 37,
    "task1_weight": 1,
    "task2_weight": 1,
    "train_batch_size": 8,
    "val_batch_size": 8,
    "model_name": "DLinkNetMTL",
    "checkpoint_path": f"{CHECKPOINT_DIR}/final_model.pth",
    "test_dir": TEST_DIR,
    "output_dir": OUTPUT_DIR,
    "nfolds": 5,
    "classification_threshold": 0.5,
    "backbone_pretrained": True,
    "train_dataset": {
        "deepglobe":{
            "dir": TRAIN_DIR,
            "crop_size": 400,
            "stride": 128,
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
            "dir": TRAIN_DIR,
            "crop_size": 400
        },
        "augmentation": False,
        "mean" : "[75.42847031, 84.17389662, 84.91022512]",
        "std" : "[47.70426159, 47.7377937, 49.46134476]",
        "normalize_type": "Mean",
        "angle_theta": 10,
        "angle_bin": 10
    },
    "optimizer": {
        "lr": 1e-3,
        "weight_decay": 0.0005,
    },
    "trainer": {
        "total_epochs": 100,
        "logging_dir": LOGGING_DIR,
    }
},

{
    "seed": 7,
    "task1_classes": 2,
    "task2_classes": 37,
    "task1_weight": 1,
    "task2_weight": 1,
    "train_batch_size": 8,
    "val_batch_size": 8,
    "model_name": "DLinkNetMTL",
    "checkpoint_path": f"{CHECKPOINT_DIR}/final_model.pth",
    "test_dir": TEST_DIR,
    "output_dir": OUTPUT_DIR,
    "nfolds": 5,
    "classification_threshold": 0.5,
    "full_frame_inference": True,
    "backbone_pretrained": True,
    "train_dataset": {
        "deepglobe":{
            "dir": TRAIN_DIR,
            "crop_size": 400,
            "stride": 32,
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
            "dir": TRAIN_DIR,
            "crop_size": 400
        },
        "augmentation": False,
        "mean" : "[75.42847031, 84.17389662, 84.91022512]",
        "std" : "[47.70426159, 47.7377937, 49.46134476]",
        "normalize_type": "Mean",
        "angle_theta": 10,
        "angle_bin": 10
    },
    "optimizer": {
        "lr": 5e-4,
        "weight_decay": 0.0005,
    },
    "trainer": {
        "total_epochs": 200,
        "logging_dir": LOGGING_DIR,
    }
},


{
    "seed": 7,
    "task1_classes": 2,
    "task2_classes": 37,
    "task1_weight": 1,
    "task2_weight": 1,
    "train_batch_size": 8,
    "val_batch_size": 8,
    "model_name": "SpinDLinkNetMTL",
    "checkpoint_path": "/content/checkpoints/final_model.pth",
    "test_dir": "/content/drive/MyDrive/CS433/test_set_images/",
    "output_dir": "/content/results/",
    "nfolds": 5,
    "classification_threshold": 0.5,
    "backbone_pretrained": True,
    "train_dataset": {
        "deepglobe":{
            "dir": "/content/drive/MyDrive/CS433/training/",
            "crop_size": 400,
            "stride": 128,
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
            "crop_size": 400
        },
        "augmentation": False,
        "mean" : "[75.42847031, 84.17389662, 84.91022512]",
        "std" : "[47.70426159, 47.7377937, 49.46134476]",
        "normalize_type": "Mean",
        "angle_theta": 10,
        "angle_bin": 10
    },
    "optimizer": {
        "lr": 1e-3,
        "weight_decay": 0.0005,
    },
    "trainer": {
        "total_epochs": 100,
        "logging_dir": "/content/logs/",
    }
},

{
    "seed": 7,
    "task1_classes": 2,
    "task2_classes": 37,
    "task1_weight": 1,
    "task2_weight": 1,
    "train_batch_size": 8,
    "val_batch_size": 8,
    "model_name": "SpinDLinkNetMTL",
    "checkpoint_path": "/content/checkpoints/final_model.pth",
    "test_dir": "/content/drive/MyDrive/CS433/test_set_images/",
    "output_dir": "/content/results/",
    "nfolds": 5,
    "classification_threshold": 0.5,
    "full_frame_inference": True,
    "backbone_pretrained": True,
    "train_dataset": {
        "deepglobe":{
            "dir": "/content/drive/MyDrive/CS433/training/",
            "crop_size": 400,
            "stride": 128,
        },
        "augmentation": True,
        "mean" : "[75.42847031, 84.17389662, 84.91022512]",
        "std" : "[47.70426159, 47.7377937, 49.46134476]",
        "normalize_type": "Std",
        "angle_theta": 10,
        "angle_bin": 10
    },
    "val_dataset": {
        "deepglobe":{
            "dir": "/content/drive/MyDrive/CS433/training/",
            "crop_size": 400
        },
        "augmentation": False,
        "mean" : "[75.42847031, 84.17389662, 84.91022512]",
        "std" : "[47.70426159, 47.7377937, 49.46134476]",
        "normalize_type": "Std",
        "angle_theta": 10,
        "angle_bin": 10
    },
    "optimizer": {
        "lr": 1e-3,
        "weight_decay": 0.0005,
    },
    "trainer": {
        "total_epochs": 200,
        "logging_dir": "/content/logs/",
    }
},

{
    "seed": 7,
    "task1_classes": 2,
    "task2_classes": 37,
    "task1_weight": 1,
    "task2_weight": 1,
    "train_batch_size": 8,
    "val_batch_size": 8,
    "model_name": "LinkNet",
    "checkpoint_path": f"{CHECKPOINT_DIR}/final_model.pth",
    "test_dir": TEST_DIR,
    "output_dir": OUTPUT_DIR,
    "nfolds": 5,
    "classification_threshold": 0.5,
    "backbone_pretrained": False,
    "train_dataset": {
        "deepglobe":{
            "dir": TRAIN_DIR,
            "crop_size": 400,
            "stride": 128,
        },
        "augmentation": False,
        "mean" : "[75.42847031, 84.17389662, 84.91022512]",
        "std" : "[47.70426159, 47.7377937, 49.46134476]",
        "normalize_type": "Mean",
        "angle_theta": 10,
        "angle_bin": 10
    },
    "val_dataset": {
        "deepglobe":{
            "dir": TRAIN_DIR,
            "crop_size": 400
        },
        "augmentation": False,
        "mean" : "[75.42847031, 84.17389662, 84.91022512]",
        "std" : "[47.70426159, 47.7377937, 49.46134476]",
        "normalize_type": "Mean",
        "angle_theta": 10,
        "angle_bin": 10
    },
    "optimizer": {
        "lr": 1e-3,
        "weight_decay": 0.0005,
    },
    "trainer": {
        "total_epochs": 200,
        "logging_dir": LOGGING_DIR,
    }
},

{
    "seed": 7,
    "task1_classes": 2,
    "task2_classes": 37,
    "task1_weight": 1,
    "task2_weight": 1,
    "train_batch_size": 8,
    "val_batch_size": 8,
    "model_name": "LinkNet",
    "checkpoint_path": f"{CHECKPOINT_DIR}/final_model.pth",
    "test_dir": TEST_DIR,
    "output_dir": OUTPUT_DIR,
    "nfolds": 5,
    "classification_threshold": 0.5,
    "full_frame_inference": True,
    "backbone_pretrained": False,
    "train_dataset": {
        "deepglobe":{
            "dir": TRAIN_DIR,
            "crop_size": 400,
            "stride": 128,
        },
        "augmentation": False,
        "mean" : "[75.42847031, 84.17389662, 84.91022512]",
        "std" : "[47.70426159, 47.7377937, 49.46134476]",
        "normalize_type": "Mean",
        "angle_theta": 10,
        "angle_bin": 10
    },
    "val_dataset": {
        "deepglobe":{
            "dir": TRAIN_DIR,
            "crop_size": 400
        },
        "augmentation": False,
        "mean" : "[75.42847031, 84.17389662, 84.91022512]",
        "std" : "[47.70426159, 47.7377937, 49.46134476]",
        "normalize_type": "Mean",
        "angle_theta": 10,
        "angle_bin": 10
    },
    "optimizer": {
        "lr": 1e-3,
        "weight_decay": 0.0005,
    },
    "trainer": {
        "total_epochs": 200,
        "logging_dir": LOGGING_DIR,
    }
},

{
    "seed": 7,
    "task1_classes": 2,
    "task2_classes": 37,
    "task1_weight": 1,
    "task2_weight": 1,
    "train_batch_size": 8,
    "val_batch_size": 8,
    "model_name": "UNet",
    "checkpoint_path": "/content/checkpoints/final_model.pth",
    "test_dir": "/content/drive/MyDrive/CS433/test_set_images/",
    "output_dir": "/content/results/",
    "nfolds": 5,
    "classification_threshold": 0.5,
    "backbone_pretrained": False,
    "train_dataset": {
        "deepglobe":{
            "dir": "/content/drive/MyDrive/CS433/training/",
            "crop_size": 400,
            "stride": 128,
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
            "crop_size": 400
        },
        "augmentation": False,
        "mean" : "[75.42847031, 84.17389662, 84.91022512]",
        "std" : "[47.70426159, 47.7377937, 49.46134476]",
        "normalize_type": "Mean",
        "angle_theta": 10,
        "angle_bin": 10
    },
    "optimizer": {
        "lr": 5e-4,
        "weight_decay": 0.0005,
    },
    "trainer": {
        "total_epochs": 200,
        "logging_dir": "/content/logs/",
    }
},

{
    "seed": 7,
    "task1_classes": 2,
    "task2_classes": 37,
    "task1_weight": 1,
    "task2_weight": 1,
    "train_batch_size": 8,
    "val_batch_size": 8,
    "model_name": "UNet",
    "checkpoint_path": "/content/checkpoints/final_model.pth",
    "test_dir": "/content/drive/MyDrive/CS433/test_set_images/",
    "output_dir": "/content/results/",
    "nfolds": 5,
    "classification_threshold": 0.5,
    "full_frame_inference": True,
    "backbone_pretrained": False,
    "train_dataset": {
        "deepglobe":{
            "dir": "/content/drive/MyDrive/CS433/training/",
            "crop_size": 400,
            "stride": 128,
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
            "crop_size": 400
        },
        "augmentation": False,
        "mean" : "[75.42847031, 84.17389662, 84.91022512]",
        "std" : "[47.70426159, 47.7377937, 49.46134476]",
        "normalize_type": "Mean",
        "angle_theta": 10,
        "angle_bin": 10
    },
    "optimizer": {
        "lr": 5e-4,
        "weight_decay": 0.0005,
    },
    "trainer": {
        "total_epochs": 200,
        "logging_dir": "/content/logs/",
    }
},

]
