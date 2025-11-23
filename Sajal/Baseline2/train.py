import os
import sys
import matplotlib.image as mpimg
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from constants import SEED, TRAIN_LABELS_FILENAME, TRAIN_DATA_FILENAME, TRAINING_SIZE, BATCH_SIZE, NUM_EPOCHS, RESTORE_MODEL, TRAIN_DIR
from utils import extract_data, extract_labels, get_prediction_with_groundtruth, get_prediction_with_overlay, error_rate
from models import RoadSegmentationCNN

# Set random seeds for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)

# PyTorch Dataset
class RoadSegmentationDataset(Dataset):
    def __init__(self, data, labels):
        # Convert to PyTorch tensors and permute dimensions from (N, H, W, C) to (N, C, H, W)
        self.data = torch.from_numpy(data).float().permute(0, 3, 1, 2)
        self.labels = torch.from_numpy(labels).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Extract data
    train_data = extract_data(TRAIN_DATA_FILENAME, TRAINING_SIZE)
    train_labels = extract_labels(TRAIN_LABELS_FILENAME, TRAINING_SIZE)

    # Balance training data
    c0 = np.sum(train_labels[:, 0] == 1)
    c1 = np.sum(train_labels[:, 1] == 1)
    print("Number of data points per class: c0 = " + str(c0) + " c1 = " + str(c1))

    print("Balancing training data...")
    min_c = min(c0, c1)
    idx0 = [i for i, j in enumerate(train_labels) if j[0] == 1]
    idx1 = [i for i, j in enumerate(train_labels) if j[1] == 1]
    new_indices = idx0[0:min_c] + idx1[0:min_c]
    print(f"New dataset size: {len(new_indices)}")
    
    train_data = train_data[new_indices, :, :, :]
    train_labels = train_labels[new_indices]

    c0 = np.sum(train_labels[:, 0] == 1)
    c1 = np.sum(train_labels[:, 1] == 1)
    print("Number of data points per class: c0 = " + str(c0) + " c1 = " + str(c1))

    # Create dataset and dataloader
    dataset = RoadSegmentationDataset(train_data, train_labels)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize model
    model = RoadSegmentationCNN().to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.0)
    
    # Learning rate scheduler
    train_size = len(dataset)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    # Load model if requested
    if RESTORE_MODEL:
        checkpoint_path = os.path.join(TRAIN_DIR, "model.pth")
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path))
            print("Model restored.")
    else:
        print("Initialized!")
        print(f"Total number of iterations = {int(NUM_EPOCHS * train_size / BATCH_SIZE)}")

        # Training loop
        global_step = 0
        for epoch in range(NUM_EPOCHS):
            model.train()
            epoch_loss = 0
            epoch_error = 0
            num_batches = 0

            for batch_idx, (data, labels) in enumerate(dataloader):
                data, labels = data.to(device), labels.to(device)

                # Forward pass
                outputs = model(data)
                
                # Convert one-hot labels to class indices for CrossEntropyLoss
                label_indices = torch.argmax(labels, dim=1)
                loss = criterion(outputs, label_indices)
                
                # L2 regularization
                l2_reg = 0
                for param in model.parameters():
                    l2_reg += torch.norm(param)
                loss += 5e-4 * l2_reg

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Calculate error rate
                predictions = torch.softmax(outputs, dim=1).detach().cpu().numpy()
                labels_np = labels.cpu().numpy()
                batch_error = error_rate(predictions, labels_np)

                epoch_loss += loss.item()
                epoch_error += batch_error
                num_batches += 1

                global_step += 1

            current_lr = optimizer.param_groups[0]['lr']
            epoch_loss = epoch_loss/len(dataloader)
            print(f"Epoch {epoch}")
            print(f"Train loss: {epoch_loss:.3f}, learning rate: {current_lr:.6f}")

            # Step the scheduler once per epoch
            scheduler.step()

            # Save model
            checkpoint_path = '/content/model.pth'
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model saved in file: {checkpoint_path}")

    # Generate predictions
    print("Running prediction on training set")
    prediction_training_dir = "predictions_training/"
    os.makedirs(prediction_training_dir, exist_ok=True)
    
    model.eval()
    for i in range(1, TRAINING_SIZE + 1):
        pimg = get_prediction_with_groundtruth(model, TRAIN_DATA_FILENAME, i, device)
        Image.fromarray(pimg).save(
            prediction_training_dir + "prediction_" + str(i) + ".png"
        )
        oimg = get_prediction_with_overlay(model, TRAIN_DATA_FILENAME, i, device)
        oimg.save(prediction_training_dir + "overlay_" + str(i) + ".png")

    print("Training complete!")


if __name__ == "__main__":
    main()