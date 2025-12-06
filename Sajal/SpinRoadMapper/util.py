import math
import os
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from skimage.morphology import skeletonize


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def setSeed(config):
    manualSeed = config["seed"]
    print("Random Seed: ", manualSeed)
    # Python random module
    random.seed(manualSeed)
    # Numpy random module
    np.random.seed(manualSeed)
    # Torch random seeds
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    # Set PYTHONHASHSEED environment variable for hash-based operations
    os.environ["PYTHONHASHSEED"] = str(manualSeed)
    # Ensure deterministic behavior in cudnn (may slow down your training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def getParllelNetworkStateDict(state_dict):
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


def to_variable(tensor, requires_grad=True):
    return Variable(tensor.long().cuda(), requires_grad=requires_grad)


def weights_init(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


def weights_normal_init(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)


def performAngleMetrics(
    train_loss_angle_file, val_loss_angle_file, epoch, hist, is_train=True, write=False
):

    pixel_accuracy = np.diag(hist).sum() / hist.sum()
    mean_accuracy = np.diag(hist) / hist.sum(1)
    mean_iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    freq = hist.sum(1) / hist.sum()
    fwavacc = (freq[freq > 0] * mean_iou[freq > 0]).sum()
    if write and is_train:
        train_loss_angle_file.write(
            "[%d], Pixel Accuracy:%.3f, Mean Accuracy:%.3f, Mean IoU:%.3f, Freq.Weighted Accuray:%.3f  \n"
            % (
                epoch,
                100 * pixel_accuracy,
                100 * np.nanmean(mean_accuracy),
                100 * np.nanmean(mean_iou),
                100 * fwavacc,
            )
        )
    elif write and not is_train:
        val_loss_angle_file.write(
            "[%d], Pixel Accuracy:%.3f, Mean Accuracy:%.3f, Mean IoU:%.3f, Freq.Weighted Accuray:%.3f  \n"
            % (
                epoch,
                100 * pixel_accuracy,
                100 * np.nanmean(mean_accuracy),
                100 * np.nanmean(mean_iou),
                100 * fwavacc,
            )
        )

    return 100 * pixel_accuracy, 100 * np.nanmean(mean_iou), 100 * fwavacc


def performMetrics(
    train_loss_file,
    val_loss_file,
    epoch,
    hist,
    loss,
    loss_vec,
    is_train=True,
    write=False,
):

    pixel_accuracy = np.diag(hist).sum() / hist.sum()
    mean_accuracy = np.diag(hist) / hist.sum(1)
    mean_iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    freq = hist.sum(1) / hist.sum()
    fwavacc = (freq[freq > 0] * mean_iou[freq > 0]).sum()

    TP = hist[1, 1]
    FP = hist[0, 1]
    FN = hist[1, 0]
    f1_class1 = (2 * TP) / (2 * TP + FP + FN + 1e-12)

    if write and is_train:
        train_loss_file.write(
            "[%d], Loss:%.5f, Loss(VecMap):%.5f, Pixel Accuracy:%.3f, Mean Accuracy:%.3f, Mean IoU:%.3f, Class IoU:[%.5f/%.5f], Freq.Weighted Accuray:%.3f  \n"
            % (
                epoch,
                loss,
                loss_vec,
                100 * pixel_accuracy,
                100 * np.nanmean(mean_accuracy),
                100 * np.nanmean(mean_iou),
                mean_iou[0],
                mean_iou[1],
                100 * fwavacc,
            )
        )
    elif write and not is_train:
        val_loss_file.write(
            "[%d], Loss:%.5f, Loss(VecMap):%.5f, Pixel Accuracy:%.3f, Mean Accuracy:%.3f, Mean IoU:%.3f, Class IoU:[%.5f/%.5f], Freq.Weighted Accuray:%.3f  \n"
            % (
                epoch,
                loss,
                loss_vec,
                100 * pixel_accuracy,
                100 * np.nanmean(mean_accuracy),
                100 * np.nanmean(mean_iou),
                mean_iou[0],
                mean_iou[1],
                100 * fwavacc,
            )
        )

    return (
        100 * pixel_accuracy,
        100 * np.nanmean(mean_iou),
        100 * mean_iou[1],
        100 * fwavacc,
        100 * f1_class1,
    )


def fast_hist(a, b, n):

    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def save_checkpoint(epoch, loss, model, optimizer, best_accuracy, best_miou, config, experiment_dir):

    if torch.cuda.device_count() > 1:
        arch = type(model.module).__name__
    else:
        arch = type(model).__name__
    state = {
        "arch": arch,
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "pixel_accuracy": best_accuracy,
        "miou": best_miou,
        "config": config,
    }
    filename = os.path.join(
        experiment_dir, "checkpoint-epoch{:03d}-loss-{:.4f}.pth.tar".format(
            epoch, loss)
    )
    torch.save(state, filename)
    os.rename(filename, os.path.join(experiment_dir, "model_best.pth.tar"))
    print("Saving current best: {} ...".format("model_best.pth.tar"))


def savePredictedProb(
    real,
    gt,
    predicted,
    predicted_prob,
    pred_affinity=None,
    image_name="",
    norm_type="Mean",
):
    b, c, h, w = real.size()
    grid = []
    mean_bgr = np.array([70.95016901, 71.16398124, 71.30953645])
    deviation_bgr = np.array([34.00087859, 35.18201658, 36.40463264])

    for idx in range(b):
        # real_ = np.asarray(real[idx].numpy().transpose(1,2,0),dtype=np.float32)
        real_ = np.asarray(real[idx].numpy().transpose(
            1, 2, 0), dtype=np.float32)
        if norm_type == "Mean":
            real_ = real_ + mean_bgr
        elif norm_type == "Std":
            real_ = (real_ * deviation_bgr) + mean_bgr

        real_ = np.asarray(real_, dtype=np.uint8)
        gt_ = gt[idx].numpy() * 255.0
        gt_ = np.asarray(gt_, dtype=np.uint8)
        gt_ = np.stack((gt_,) * 3).transpose(1, 2, 0)

        predicted_ = (predicted[idx]).numpy() * 255.0
        predicted_ = np.asarray(predicted_, dtype=np.uint8)
        predicted_ = np.stack((predicted_,) * 3).transpose(1, 2, 0)

        predicted_prob_ = (predicted_prob[idx]).numpy() * 255.0
        # predicted_prob_ = predicted_prob_[:,:]
        predicted_prob_ = np.asarray(predicted_prob_, dtype=np.uint8)
        # predicted_prob_ = np.stack((predicted_prob_,)*3).transpose(1,2,0)
        predicted_prob_ = cv2.applyColorMap(predicted_prob_, cv2.COLORMAP_JET)

        if pred_affinity is not None:
            hsv = np.zeros_like(real_)
            hsv[..., 1] = 255
            affinity_ = pred_affinity[idx].numpy()
            mag = np.copy(affinity_)
            mag[mag < 36] = 1
            mag[mag >= 36] = 0
            affinity_[affinity_ == 36] = 0

            # mag, ang = cv2.cartToPolar(affinity_[0], affinity_[1])
            hsv[..., 0] = affinity_ * 10 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            affinity_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            pair = np.concatenate(
                (real_, gt_, predicted_, predicted_prob_, affinity_bgr), axis=1
            )
        else:
            pair = np.concatenate(
                (real_, gt_, predicted_, predicted_prob_), axis=1)
        grid.append(pair)

    if pred_affinity is not None:
        cv2.imwrite(image_name, np.array(grid).reshape(b * h, 5 * w, 3))
    else:
        cv2.imwrite(image_name, np.array(grid).reshape(b * h, 4 * w, 3))


def get_relaxed_precision(a, b, buffer):
    tp = 0
    indices = np.where(a == 1)
    for ind in range(len(indices[0])):
        tp += (np.sum(
            b[indices[0][ind]-buffer: indices[0][ind]+buffer+1,
              indices[1][ind]-buffer: indices[1][ind]+buffer+1]) > 0).astype(np.int)
    return tp


def relaxed_f1(pred, gt, buffer=3):
    ''' Usage and Call
    # rp_tp, rr_tp, pred_p, gt_p = relaxed_f1(predicted.cpu().numpy(), labels.cpu().numpy(), buffer = 3)

    # rprecision_tp += rp_tp
    # rrecall_tp += rr_tp
    # pred_positive += pred_p
    # gt_positive += gt_p

    # precision = rprecision_tp/(gt_positive + 1e-12)
    # recall = rrecall_tp/(gt_positive + 1e-12)
    # f1measure = 2*precision*recall/(precision + recall + 1e-12)
    # iou = precision*recall/(precision+recall-(precision*recall) + 1e-12)
    '''

    rprecision_tp, rrecall_tp, pred_positive, gt_positive = 0, 0, 0, 0
    for b in range(pred.shape[0]):
        pred_sk = skeletonize(pred[b])
        gt_sk = skeletonize(gt[b])
        # pred_sk = pred[b]
        # gt_sk = gt[b]
        rprecision_tp += get_relaxed_precision(pred_sk, gt_sk, buffer)
        rrecall_tp += get_relaxed_precision(gt_sk, pred_sk, buffer)
        pred_positive += len(np.where(pred_sk == 1)[0])
        gt_positive += len(np.where(gt_sk == 1)[0])

    return rprecision_tp, rrecall_tp, pred_positive, gt_positive

def binarize(img, threshold=128):
    """
    Binarize a grayscale image array so values become either 0 or 255.

    Parameters:
        img (np.ndarray): Input image (grayscale or any numeric array).
        threshold (int or float): Threshold value.

    Returns:
        np.ndarray: Binarized image.
    """
    return np.where(img >= threshold, 255.0, 0.0)

def compute_dataset_mean_std():
    folder = "/content/drive/MyDrive/CS433/training/images/"

    sum_c = np.zeros(3, dtype=np.float64)
    sum_sq_c = np.zeros(3, dtype=np.float64)
    count = 0

    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))

        # Normalize to [0,1] or keep in [0,255] depending on your convention
        img = img.astype(np.float32)

        # Compute mean and std for this image
        sum_c += img.reshape(-1, 3).sum(axis=0)
        sum_sq_c += (img.reshape(-1, 3) ** 2).sum(axis=0)
        num_pixels = img.shape[0] * img.shape[1]
        count += num_pixels

    # Per-channel mean
    mean = sum_c / count

    # Per-channel std
    std = np.sqrt(sum_sq_c / count - mean**2)

    print("Mean (R, G, B):", mean)
    print("Std  (R, G, B):", std)