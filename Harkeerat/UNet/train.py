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


def dice_coef(y_true, y_pred):
    """
    Calculates the F1 Score (Dice Coefficient) for tracking.
    """
    smooth = 1.0
    # Flatten the arrays
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )
    return score


def train_model():
    print("--- Starting Training Pipeline ---")

    # 1. Load Data
    X, Y = load_data(config.TRAIN_IMAGES_PATH, config.TRAIN_MASKS_PATH)

    # 2. Setup Cross Validation
    kfold = KFold(n_splits=config.NUM_FOLDS, shuffle=True, random_state=42)
    fold_no = 1

    # List to store F1 scores for each fold
    f1_scores = []

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
            metrics=["accuracy", dice_coef],
        )

        # Checkpoint: Save only the best model for this fold
        save_path = os.path.join(config.MODEL_SAVE_DIR, f"unet_fold_{fold_no}.keras")
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            save_path,
            save_best_only=True,
            monitor="val_dice_coef",
            mode="max",
            verbose=1,
        )

        # Train
        history = model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            validation_data=(X_val, Y_val),
            epochs=config.EPOCHS,
            callbacks=[checkpoint],
        )

        # Evaluate the model on the validation set
        val_metrics = model.evaluate(X_val, Y_val, verbose=0)
        val_f1_score = val_metrics[2]  # Assuming dice_coef is the third metric
        print(f"Fold {fold_no} F1 Score (Dice Coefficient): {val_f1_score:.4f}")
        f1_scores.append(val_f1_score)

        fold_no += 1

    # Calculate and print the average F1 score
    avg_f1_score = sum(f1_scores) / len(f1_scores)
    print(
        f"\nAverage F1 Score (Dice Coefficient) across {config.NUM_FOLDS} folds: {avg_f1_score:.4f}"
    )


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
        metrics=["accuracy", dice_coef],
    )

    # 4. Checkpoint (Saves the BEST model based on lowest loss)
    best_save_path = os.path.join(config.MODEL_SAVE_DIR, "unet_full_best.keras")
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        best_save_path, save_best_only=True, monitor="dice_coef", mode="max", verbose=1
    )

    print(f"Training on {len(X)} images for {config.EPOCHS} epochs...")

    # 5. Train
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=config.EPOCHS,
        callbacks=[checkpoint],
    )
    
    # Evaluate the model on the training set
    train_metrics = model.evaluate(X, Y, verbose=0)
    train_f1_score = train_metrics[2]  # Assuming dice_coef is the third metric
    print(f"Full Training F1 Score (Dice Coefficient): {train_f1_score:.4f}")

    # 6. EXPLICITLY Save the Final Model (Last Epoch)
    final_save_path = os.path.join(config.MODEL_SAVE_DIR, "unet_full_final.keras")
    print(f"Saving final model to {final_save_path}...")
    model.save(final_save_path)

    print("Done!")


if __name__ == "__main__":
    train_model()
    train_full_model()
