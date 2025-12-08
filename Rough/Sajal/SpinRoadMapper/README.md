Make sure the config.py has the correct checkpoint_path, test_dir, output_dir, logging_dir.

## To run cross-validation (with config 0):
```
python3 train.py --crossval 1 --config 0
```

## To run training (with config 0):
```
python3 train.py --crossval 0 --config 0
```

## To run inference:
```
python3 inference.py
```
