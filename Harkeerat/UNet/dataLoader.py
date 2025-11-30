import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_data(image_path, mask_path=None):
    """
    Loads images and optionally masks from the specified directories.
    Normalizes images to [0, 1].
    Binarizes masks to strictly {0, 1}.
    """
    images = []
    masks = []

    # Get sorted list of filenames to ensure alignment
    # We filter for .png or .jpg to avoid reading hidden system files
    img_files = sorted(
        [f for f in os.listdir(image_path) if f.endswith((".png", ".jpg", ".jpeg"))]
    )

    print(f"Loading {len(img_files)} images from {image_path}...")

    for filename in img_files:
        # Load Image
        img = cv2.imread(os.path.join(image_path, filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        img = img / 255.0  # Normalize to 0-1
        images.append(img)

        # Load Mask (if path provided)
        if mask_path:
            # Assumes mask filename matches image filename
            mask = cv2.imread(os.path.join(mask_path, filename), cv2.IMREAD_GRAYSCALE)

            # --- CRITICAL FIX ---
            # Instead of dividing by 255, we THRESHOLD.
            # Any pixel > 127 (gray) becomes 1.0 (Road), others 0.0 (Background).
            # This handles values like 237, 255, or 1 correctly.
            mask = (mask > 127).astype(np.float32)

            mask = np.expand_dims(mask, axis=-1)  # (400, 400) -> (400, 400, 1)
            masks.append(mask)

    if mask_path:
        return np.array(images), np.array(masks)
    else:
        return np.array(images), None


def get_augmented_train_generator(X_train, Y_train, batch_size, seed=42):
    """
    Returns a synchronized generator for image and mask augmentation.
    """
    aug_dict = dict(
        rotation_range=25,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,  # Safe for aerial maps
        fill_mode="reflect",  # Reflect handles corners better than 'constant'
    )

    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)

    # Seed is critical here to ensure image and mask are transformed identically
    image_generator = image_datagen.flow(X_train, batch_size=batch_size, seed=seed)
    mask_generator = mask_datagen.flow(Y_train, batch_size=batch_size, seed=seed)

    # FIX: Newer Keras versions do not support 'zip' objects directly.
    # We must wrap it in a standard Python generator loop.
    while True:
        X_batch = next(image_generator)
        Y_batch = next(mask_generator)

        # Extra safety: ensure augmented masks stay binary 0 or 1
        # Interpolation during rotation can create values like 0.5.
        Y_batch = (Y_batch > 0.5).astype(np.float32)

        yield (X_batch, Y_batch)
