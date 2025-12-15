import os
import re
import numpy as np
import tensorflow as tf
import cv2

# Import local modules
import config
from dataLoader import load_data
from helperFunctions import mask_to_submission_strings


def generate_submission():
    print("--- Starting Prediction Pipeline ---")

    # 1. Load Test Images
    # Note: We don't have masks for test data, so second return is ignored
    test_images, _ = load_data(config.TEST_IMAGES_PATH, None)

    # Get filenames to extract IDs later
    test_files = sorted(
        [f for f in os.listdir(config.TEST_IMAGES_PATH) if f.endswith((".png", ".jpg"))]
    )

    # 2. Ensemble Prediction
    print(f"Predicting on {len(test_images)} images using {config.NUM_FOLDS} models...")

    # Container for aggregated predictions
    final_preds = np.zeros((len(test_images), config.IMG_HEIGHT, config.IMG_WIDTH, 1))

    for fold in range(1, config.NUM_FOLDS + 1):
        model_path = os.path.join(config.MODEL_SAVE_DIR, f"unet_fold_{fold}.keras")

        if not os.path.exists(model_path):
            print(
                f"Warning: Model for fold {fold} not found at {model_path}. Skipping."
            )
            continue

        print(f"Loading model: {model_path}")
        model = tf.keras.models.load_model(model_path)

        # Predict
        preds = model.predict(test_images, batch_size=config.BATCH_SIZE, verbose=1)
        final_preds += preds

    # Average the predictions across folds
    final_preds /= config.NUM_FOLDS

    # 3. Generate Submission CSV
    print(f"Writing results to {config.SUBMISSION_FILE}...")

    with open(config.SUBMISSION_FILE, "w") as f:
        f.write("id,prediction\n")

        for idx, filename in enumerate(test_files):
            # Extract ID using regex (e.g., "test_1.png" -> 1)
            # This handles various filename formats robustly
            try:
                img_number = int(re.search(r"\d+", filename).group(0))
            except AttributeError:
                print(f"Error parsing ID from {filename}, using index {idx+1}")
                img_number = idx + 1

            current_mask = final_preds[idx]

            # Use the logic from utils.py to convert mask to CSV lines
            for line in mask_to_submission_strings(current_mask, img_number):
                f.write(line + "\n")

    print("Done! Submission file created.")


if __name__ == "__main__":
    generate_submission()
