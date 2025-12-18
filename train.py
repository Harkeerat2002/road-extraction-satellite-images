import os
from datetime import datetime
import numpy as np
import torch
import torch.optim as optim
from models import MODELS
import torch.utils.data as data
from torch.autograd import Variable
import torch.nn.functional as F
import util
from loss import mIoULoss, CrossEntropyLoss2d
from dataset import DeepGlobeDataset
import argparse
from configs import configs
from torch.optim.lr_scheduler import CosineAnnealingLR

def train(epoch,
          model,
          train_loader,
          road_loss,
          angle_loss,
          optimizer,
          config, 
          mtl
          ):
    train_loss_iou = 0
    train_loss_vec = 0

    # Set model to training mode
    model.train()

    # Clear gradients
    optimizer.zero_grad()

    # Initialize confusion matrix for road segmentation and orientation tasks
    hist = np.zeros((config["task1_classes"], config["task1_classes"]))
    hist_angles = np.zeros((config["task2_classes"], config["task2_classes"]))
    crop_size = config["train_dataset"]["deepglobe"]["crop_size"]

    for i, data in enumerate(train_loader, 0):
        # Multi-Task Learning flow
        if mtl:
            # Get the inputs (images and orientation vectors) and labels
            inputsBGR, labels, vecmap_angles = data
            inputsBGR = Variable(inputsBGR.float().cuda())
            outputs, pred_vecmaps = model(inputsBGR)
            
            # Calculate losses for both tasks
            loss1 = road_loss(outputs, util.to_variable(labels[-1], requires_grad = False), False)
            loss2 = angle_loss(pred_vecmaps, util.to_variable(vecmap_angles[-1], requires_grad = False))

            train_loss_iou += loss1.item()
            train_loss_vec += loss2.item()

            # Get predictions for road segmentation on 0.5 threshold
            _, predicted = torch.max(outputs.data, 1)

            correctLabel = labels[-1].view(-1, crop_size, crop_size).long()

            # Update confusion matrix for road segmentation
            hist += util.fast_hist(
                predicted.view(predicted.size(0), -1).cpu().numpy(),
                correctLabel.view(correctLabel.size(0), -1).cpu().numpy(),
                config["task1_classes"],
            )

            # Get predictions for orientation angles and update confusion matrix
            _, predicted_angle = torch.max(pred_vecmaps.data, 1)
            correct_angles = vecmap_angles[-1].view(-1, crop_size, crop_size).long()
            hist_angles += util.fast_hist(
                predicted_angle.view(predicted_angle.size(0), -1).cpu().numpy(),
                correct_angles.view(correct_angles.size(0), -1).cpu().numpy(),
                config["task2_classes"],
            )

            # Backpropagate combined loss
            torch.autograd.backward([loss1, loss2])

            # Update model parameters
            optimizer.step()
            optimizer.zero_grad()

            del (
                outputs,
                pred_vecmaps,
                predicted,
                correct_angles,
                correctLabel,
                inputsBGR,
                labels,
                vecmap_angles,
            )
        else:
            inputsBGR, labels, vecmap_angles = data
            inputsBGR = Variable(inputsBGR.float().cuda())
            outputs = model(inputsBGR)
            
            # Calculate loss for road segmentation task only
            loss1 = road_loss(outputs, util.to_variable(labels[-1], requires_grad = False), False)

            train_loss_iou += loss1.item()

            # Get predictions for road segmentation on 0.5 threshold
            _, predicted = torch.max(outputs.data, 1)

            correctLabel = labels[-1].view(-1, crop_size, crop_size).long()

            # Update confusion matrix for road segmentation
            hist += util.fast_hist(
                predicted.view(predicted.size(0), -1).cpu().numpy(),
                correctLabel.view(correctLabel.size(0), -1).cpu().numpy(),
                config["task1_classes"],
            )

            # Backpropagate loss
            torch.autograd.backward([loss1])

            # Update model parameters
            optimizer.step()
            optimizer.zero_grad()

            del (
                outputs,
                predicted,
                correctLabel,
                inputsBGR,
                labels,
                vecmap_angles,
            )


    # Calculate and log metrics for road segmentation task
    util.performMetrics(
            train_loss_file,
            val_loss_file,
            epoch,
            hist,
            train_loss_iou / len(train_loader),
            train_loss_vec / len(train_loader),
            write=True,
    )

    # Calculate and log metrics for orientation angle task
    util.performAngleMetrics(
        train_loss_angle_file, 
        val_loss_angle_file, 
        epoch, 
        hist_angles, 
        write=True
    )


def val(epoch,
        model,
        val_loader,
        road_loss,
        angle_loss,
        fold,
        config,
        mtl
        ):

    # Set model to evaluation mode
    model.eval()

    test_loss_iou = 0
    test_loss_vec = 0

    # Initialize confusion matrix for road segmentation and orientation tasks
    hist = np.zeros((config["task1_classes"], config["task1_classes"]))
    hist_angles = np.zeros((config["task2_classes"], config["task2_classes"]))
    crop_size = config["val_dataset"]["deepglobe"]["crop_size"]

    with torch.no_grad():
        for i, (inputsBGR, labels, vecmap_angles) in enumerate(val_loader, 0):
            if mtl:
                inputsBGR = Variable(
                    inputsBGR.float().cuda(), requires_grad=False
                )
                # Get model outputs for both tasks
                outputs, pred_vecmaps = model(inputsBGR)

                # Calculate losses for both tasks
                loss1 = road_loss(outputs, util.to_variable(labels[0], requires_grad = False), True)
                loss2 = angle_loss(pred_vecmaps, util.to_variable(labels[0], requires_grad = False))

                test_loss_iou += loss1.item()
                test_loss_vec += loss2.item()

                # Get predictions for road segmentation on 0.5 threshold
                _, predicted = torch.max(outputs.data, 1)

                # Update confusion matrix for road segmentation
                correctLabel = labels[-1].view(-1, crop_size, crop_size).long()
                hist += util.fast_hist(
                    predicted.view(predicted.size(0), -1).cpu().numpy(),
                    correctLabel.view(correctLabel.size(0), -1).cpu().numpy(),
                    config["task1_classes"],
                )

                # Get predictions for orientation angles and update confusion matrix
                _, predicted_angle = torch.max(pred_vecmaps.data, 1)
                correct_angles = vecmap_angles[-1].view(-1, crop_size, crop_size).long()
                hist_angles += util.fast_hist(
                    predicted_angle.view(predicted_angle.size(0), -1).cpu().numpy(),
                    correct_angles.view(correct_angles.size(0), -1).cpu().numpy(),
                    config["task2_classes"],
                )

                # Save validation images for the last epoch
                if epoch == config["trainer"]["total_epochs"] and i == len(val_loader) - 1:
                    images_path = os.path.join(experiment_dir, "images", str(fold))
                    os.makedirs(images_path, exist_ok=True)
                    util.savePredictedProb(
                        inputsBGR.data.cpu(),
                        labels[-1].cpu(),
                        predicted.cpu(),
                        F.softmax(outputs, dim=1).data.cpu()[:, 1, :, :],
                        predicted_angle.cpu(),
                        os.path.join(images_path, "validate_pair_{}_{}.png".format(epoch, i)),
                        norm_type=config["val_dataset"]["normalize_type"],
                    )

                del inputsBGR, labels, predicted, outputs, pred_vecmaps, predicted_angle

            else:
                inputsBGR = Variable(
                    inputsBGR.float().cuda(), requires_grad=False
                )

                # Get model outputs for road segmentation task only
                outputs = model(inputsBGR)

                # Calculate loss for road segmentation task only
                loss1 = road_loss(outputs, util.to_variable(labels[0], requires_grad = False), True)

                test_loss_iou += loss1.item()

                # Get predictions for road segmentation on 0.5 threshold
                _, predicted = torch.max(outputs.data, 1)

                # Update confusion matrix for road segmentation
                correctLabel = labels[-1].view(-1, crop_size, crop_size).long()
                hist += util.fast_hist(
                    predicted.view(predicted.size(0), -1).cpu().numpy(),
                    correctLabel.view(correctLabel.size(0), -1).cpu().numpy(),
                    config["task1_classes"],
                )

                del inputsBGR, labels, predicted, outputs

    # Calculate and log metrics for road segmentation task
    accuracy, miou, road_iou, fwacc, f1class1 = util.performMetrics(
        train_loss_file,
        val_loss_file,
        epoch,
        hist,
        test_loss_iou / len(val_loader),
        test_loss_vec / len(val_loader),
        is_train=False,
        write=True,
    )
    # Calculate and log metrics for orientation angle task
    util.performAngleMetrics(
        train_loss_angle_file,
        val_loss_angle_file,
        epoch,
        hist_angles,
        is_train=False,
        write=True,
    )
    # Return validation loss and F1 score for road class
    return test_loss_iou / len(val_loader), f1class1



def train_fold(config, fold, n_folds):
    print(f"\n{'='*50}")
    print(f"Training Fold {fold + 1}/{n_folds}")
    print(f"{'='*50}\n")

    # Set random seed for reproducibility
    util.setSeed(config)

    # Initialize model
    MODEL_CHOOSEN = config["model_name"]

    # Multi-task learning flag
    if MODEL_CHOOSEN in ['DLinkNetMTL', 'SpinDLinkNetMTL']:
        mtl = True
        model = MODELS[MODEL_CHOOSEN](
        config["task1_classes"], config["task2_classes"], config["backbone_pretrained"]
        )
        print("Multi-task learning enabled(road segmentation and orientation learning).")
    else:
        mtl = False
        model = MODELS[MODEL_CHOOSEN](
        config["task1_classes"], config["backbone_pretrained"]
    )
        print("Single-task learning (road segmentation only).")
    # Move model to GPU
    model.cuda()

    # Initialize model weights if not using pretrained backbone
    if config["backbone_pretrained"]:
        print("Using pretrained backbone weights.")
    else:
        util.weights_init(model)

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), 
                                 lr=config["optimizer"]["lr"],
                                 weight_decay=config["optimizer"]["weight_decay"])

    # Initialize LR scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=config["trainer"]["total_epochs"])

    # Initialize losses
    weights = torch.ones(config["task1_classes"]).cuda()
    if config["task1_weight"] < 1:
        print("Roads are weighted.")
        weights[0] = 1 - config["task1_weight"]
        weights[1] = config["task1_weight"]

    road_loss = mIoULoss(
        weight=weights, size_average=True, n_classes=config["task1_classes"]
    ).cuda()

    weights_angles = torch.ones(config["task2_classes"]).cuda()
    if config["task2_weight"] < 1:
        print("Road angles are weighted.")
        weights_angles[-1] = config["task2_weight"]

    angle_loss = CrossEntropyLoss2d(
        weight=weights_angles, size_average=True, ignore_index=255, reduce=True
    ).cuda()

    # Initialize validation and training variables
    f1score = 0
    start_epoch = 1
    total_epochs = config["trainer"]["total_epochs"]

    # Initialize data loaders
    g = torch.Generator()
    g.manual_seed(config["seed"])

    train_loader = data.DataLoader(
    DeepGlobeDataset(
        config["train_dataset"],
        seed=config["seed"],
        is_train=True,
        fold=fold,
        n_folds=n_folds
    ),
    batch_size=config["train_batch_size"],
    shuffle=True,
    generator=g,
    )
    # Initialize validation data loader only if using cross-validation
    if n_folds != 1:
        val_loader = data.DataLoader(
        DeepGlobeDataset(
            config["val_dataset"],
            seed=config["seed"],
            is_train=False,
            fold=fold,
            n_folds=n_folds
        ),
        batch_size=config["val_batch_size"],
        shuffle=False,
        )
        print(f"Fold {fold + 1}: Train size = {len(train_loader.dataset)}, Val size = {len(val_loader.dataset)}")
    else:
        print("Training on the entire dataset without cross-validation.")
        print(f"Fold {fold + 1}: Train size = {len(train_loader.dataset)}")
        

    # Start training
    for epoch in range(start_epoch, total_epochs + 1):
        print("\nTraining Epoch: %d" % epoch)
        train(epoch,
              model,
              train_loader,
              road_loss,
              angle_loss,
              optimizer,
              config,
              mtl
              )

        if n_folds != 1:
            print("\nTesting Epoch: %d" % epoch)
            
            _, f1score = val(epoch,
                            model,
                            val_loader,
                            road_loss,
                            angle_loss,
                            fold,
                            config,
                            mtl
                            )
            print(f"Epoch {epoch} completed, f1score: {f1score:.4f}")
        # Update learning rate
        scheduler.step()

    if n_folds != 1:
        print(f"Fold {fold + 1}, LastF1Score {f1score} Training Completed.\n{'='*50}\n")
    else:   
        print("Saving final model...")
        torch.save(model.state_dict(), config["checkpoint_path"])
        print(f"Model saved to {config['checkpoint_path']}")

    # Return validation F1 score for road class at last epoch
    return f1score
    

def cross_validation(config):
    """Perform k-fold cross-validation"""

    n_folds = config['nfolds']
    
    fold_results = []
    
    for fold in range(n_folds):
        result = train_fold(config, fold, n_folds)
        fold_results.append(result)
    
    # Calculate average metrics
    avg_val_f1score = np.mean(fold_results)
    std_val_f1score = np.std(fold_results)
    
    print(f"\n{'='*50}")
    print(f"Cross-Validation Results ({n_folds} folds)")
    print(f"{'='*50}")
    print(f"Average Validation F1 Score: {avg_val_f1score:.4f} ± {std_val_f1score:.4f}")
    print(f"{'='*50}\n")

def train_single(config):
    """Train on the entire dataset without cross-validation"""
    os.makedirs(os.path.dirname(config["checkpoint_path"]), exist_ok=True)
    train_fold(config, fold=0, n_folds=1)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--crossval', type=int, default=1)
    parser.add_argument('--config', type=int, default=0)
    args = parser.parse_args()
    config_number = args.config
    config = configs[config_number]
    # Create experiment directory
    experiment_dir = config["trainer"]["logging_dir"]
    os.makedirs(experiment_dir, exist_ok=True)
    # Create files to log training and validation losses
    train_loss_file = open(os.path.join(experiment_dir, 'train.txt'), "w+")
    val_loss_file = open(os.path.join(experiment_dir, 'val.txt'), "w+")
    train_loss_angle_file = open(os.path.join(experiment_dir, "train_angle.txt"), "w+")
    val_loss_angle_file = open(os.path.join(experiment_dir, "val_angle.txt"), "w+")
    # Start training or cross-validation
    if args.crossval == 0:
        train_single(config)
    else:   
        cross_validation(config)