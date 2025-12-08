from constants import IMG_PATCH_SIZE, NUM_CHANNELS, NUM_LABELS
import torch.nn as nn

# CNN Model
class RoadSegmentationCNN(nn.Module):
    def __init__(self):
        super(RoadSegmentationCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(NUM_CHANNELS, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        fc1_input_size = int(IMG_PATCH_SIZE / 4 * IMG_PATCH_SIZE / 4 * 64)
        self.fc1 = nn.Linear(fc1_input_size, 512)
        self.fc2 = nn.Linear(512, NUM_LABELS)
        
        # Activation
        self.relu = nn.ReLU()

    def forward(self, x):
        # First conv block
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Flatten
        x = x.reshape(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x