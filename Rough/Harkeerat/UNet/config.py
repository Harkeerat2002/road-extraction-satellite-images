import os

# --- Paths ---
# Ensure your folder structure matches these or change them here
TRAIN_IMAGES_PATH = "../../Dataset/training/images/"
TRAIN_MASKS_PATH = "../../Dataset/training/groundtruth/"
TEST_IMAGES_PATH = "../../Dataset/test_set_images/test/"
SUBMISSION_FILE = "submission.csv"
MODEL_SAVE_DIR = "models/"

# --- Image Parameters ---
IMG_WIDTH = 400
IMG_HEIGHT = 400
IMG_CHANNELS = 3

# --- Hyperparameters ---
NUM_FOLDS = 5  # Number of Cross-Validation Folds
BATCH_SIZE = 4  # Lower this if you run out of GPU memory
EPOCHS = 50  # Number of training epochs per fold
LEARNING_RATE = 1e-4

# --- Post-Processing ---
FOREGROUND_THRESHOLD = 0.25  # Threshold to decide if a 16x16 patch is a road
PATCH_SIZE = 16

# Create model directory if it doesn't exist
if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)
