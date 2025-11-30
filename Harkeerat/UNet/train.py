import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = (
    "3"  # 0=All, 1=Filter Info, 2=Filter Warning, 3=Filter Error
)

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold

# Import local modules
import config
from unetModel import build_unet
from dataLoader import load_data, get_augmented_train_generator


def train_model():
    print("--- Starting Training Pipeline ---")

    # 1. Load Data
    X, Y = load_data(config.TRAIN_IMAGES_PATH, config.TRAIN_MASKS_PATH)

    # 2. Setup Cross Validation
    kfold = KFold(n_splits=config.NUM_FOLDS, shuffle=True, random_state=42)
    fold_no = 1

    # 3. Training Loop
    for train_index, val_index in kfold.split(X):
        print(f"\nTraining Fold {fold_no} / {config.NUM_FOLDS}")

        # Split Data
        X_train, X_val = X[train_index], X[val_index]
        Y_train, Y_val = Y[train_index], Y[val_index]

        # Create Augmented Generator
        train_gen = get_augmented_train_generator(X_train, Y_train, config.BATCH_SIZE)
        steps_per_epoch = len(X_train) // config.BATCH_SIZE

        # Build Model
        model = build_unet((config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS))
        model.compile(
            optimizer=Adam(learning_rate=config.LEARNING_RATE),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        # Checkpoint: Save only the best model for this fold
        save_path = os.path.join(config.MODEL_SAVE_DIR, f"unet_fold_{fold_no}.keras")
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            save_path, save_best_only=True, monitor="val_loss", mode="min", verbose=1
        )

        # Train
        history = model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            validation_data=(X_val, Y_val),
            epochs=config.EPOCHS,
            callbacks=[checkpoint],
        )

        fold_no += 1


def train_full_model():
    print("--- Starting Full Training Pipeline (No Validation) ---")

    # 1. Load ALL Data
    X, Y = load_data(config.TRAIN_IMAGES_PATH, config.TRAIN_MASKS_PATH)

    # 2. Create Augmented Generator with ALL data
    # We use the entire dataset for training
    train_gen = get_augmented_train_generator(X, Y, config.BATCH_SIZE)
    steps_per_epoch = len(X) // config.BATCH_SIZE

    # 3. Build Model
    model = build_unet((config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS))
    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    # 4. Checkpoint (Saves the BEST model based on lowest loss)
    best_save_path = os.path.join(config.MODEL_SAVE_DIR, "unet_full_best.keras")
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        best_save_path, save_best_only=True, monitor="loss", mode="min", verbose=1
    )

    print(f"Training on {len(X)} images for {config.EPOCHS} epochs...")

    # 5. Train
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=config.EPOCHS,
        callbacks=[checkpoint],
    )

    # 6. EXPLICITLY Save the Final Model (Last Epoch)
    final_save_path = os.path.join(config.MODEL_SAVE_DIR, "unet_full_final.keras")
    print(f"Saving final model to {final_save_path}...")
    model.save(final_save_path)

    print("Done!")


if __name__ == "__main__":
    train_model()
    train_full_model()
