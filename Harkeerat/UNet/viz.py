import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = (
    "3"  # 0=All, 1=Filter Info, 2=Filter Warning, 3=Filter Error
)
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import config


def visualize_prediction(model_path, image_index=0):
    """
    Loads a model and visualizes its prediction on a specific training image.
    """
    # 1. Get file paths
    img_files = sorted(
        [
            f
            for f in os.listdir(config.TRAIN_IMAGES_PATH)
            if f.endswith((".png", ".jpg"))
        ]
    )
    mask_files = sorted(
        [f for f in os.listdir(config.TRAIN_MASKS_PATH) if f.endswith((".png", ".jpg"))]
    )

    if image_index >= len(img_files):
        print(
            f"Error: Index {image_index} is out of bounds. Only {len(img_files)} images found."
        )
        return

    img_name = img_files[image_index]
    mask_name = mask_files[image_index]

    print(f"Visualizing Image #{image_index}: {img_name}")

    # 2. Load and Preprocess Single Image
    img_path = os.path.join(config.TRAIN_IMAGES_PATH, img_name)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_normalized = img / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)

    # 3. Load Ground Truth Mask with Thresholding (Same as Data Loader)
    mask_path = os.path.join(config.TRAIN_MASKS_PATH, mask_name)
    true_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # --- FIX: Force clean binary mask ---
    # Any pixel > 127 becomes 1.0, else 0.0
    true_mask = (true_mask > 127).astype(np.float32)

    # 4. Load Model and Predict
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Did you run train_full.py?")
        return

    print("Loading model...")
    model = tf.keras.models.load_model(model_path)

    print("Predicting...")
    pred_mask = model.predict(img_input, verbose=0)[0]
    pred_mask = np.squeeze(pred_mask)

    # 5. Plotting
    plt.figure(figsize=(15, 5))

    # Original Image
    plt.subplot(1, 3, 1)
    plt.title("Original Satellite Image")
    plt.imshow(img_normalized)
    plt.axis("off")

    # Ground Truth
    plt.subplot(1, 3, 2)
    plt.title("True Road Mask (Thresholded)")
    # vmin=0, vmax=1 ensures 0 is black, 1 is white
    plt.imshow(true_mask, cmap="gray", vmin=0, vmax=1)
    plt.axis("off")

    # Prediction
    plt.subplot(1, 3, 3)
    plt.title("Predicted Road Mask")
    plt.imshow(pred_mask, cmap="gray", vmin=0, vmax=1)
    plt.axis("off")

    output_file = "prediction_viz.png"
    plt.savefig(output_file)
    print(f"Visualization saved to {output_file}")

    try:
        plt.show()
    except:
        pass


if __name__ == "__main__":
    # Ensure this points to your trained model
    model_file = os.path.join(config.MODEL_SAVE_DIR, "unet_full_best.keras")

    # Change index to see different images
    visualize_prediction(model_file, image_index=1)
