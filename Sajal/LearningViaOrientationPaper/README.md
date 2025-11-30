Make sure the config.py has the correct checkpoint_path, test_dir, output_dir, logging_dir.

## To run cross-validation:
```
python3 train.py --crossval 1
```

## To run training:
```
python3 train.py --crossval 0
```

## To run inference:
```
python3 inference.py
```
