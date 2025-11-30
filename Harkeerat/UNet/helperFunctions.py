import numpy as np
import config

# --- Logic adapted from user request ---


def patch_to_label(patch):
    """
    Calculates the mean of a patch. If > threshold, returns 1 (Road), else 0.
    """
    df = np.mean(patch)
    if df > config.FOREGROUND_THRESHOLD:
        return 1
    else:
        return 0


def mask_to_submission_strings(prediction_mask, img_number):
    """
    Iterates over the prediction mask in 16x16 patches and yields the CSV strings.
    prediction_mask: numpy array (Height, Width)
    """
    # Ensure 2D array
    if len(prediction_mask.shape) == 3:
        prediction_mask = np.squeeze(prediction_mask)

    height, width = prediction_mask.shape
    patch_size = config.PATCH_SIZE

    # The order of loops (j then i) is specific to the submission format required
    for j in range(0, width, patch_size):
        for i in range(0, height, patch_size):
            patch = prediction_mask[i : i + patch_size, j : j + patch_size]
            label = patch_to_label(patch)
            yield ("{:03d}_{}_{},{}".format(img_number, j, i, label))
