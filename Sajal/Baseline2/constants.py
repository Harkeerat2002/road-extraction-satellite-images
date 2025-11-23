# Hyperparameters
NUM_CHANNELS = 3  # RGB images
PIXEL_DEPTH = 255
NUM_LABELS = 2
TRAINING_SIZE = 20
VALIDATION_SIZE = 5  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 16
NUM_EPOCHS = 10
RESTORE_MODEL = False  # If True, restore existing model instead of training a new one
IMG_PATCH_SIZE = 16  # Should be a multiple of 4

# Directories
TRAIN_DIR = "/home/chaurasi/cs433/Project2/baseline2/project-2-overfit-squad-1/Sajal/Baseline2/"
DATA_DIR = "/home/chaurasi/cs433/Project2/data/training/"
TRAIN_DATA_FILENAME = DATA_DIR + "images/"
TRAIN_LABELS_FILENAME = DATA_DIR + "groundtruth/"
CHECKPOINT_DIR = TRAIN_DIR + "checkpoints/"