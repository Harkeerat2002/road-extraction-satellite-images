import torch
import numpy as np
from PIL import Image
import os
from configs import configs
from stack_module import StackHourglassNetMTL_DGCNv4
from dataset import RoadDataset
import cv2
import argparse

class InferenceEngine(RoadDataset):
    def __init__(self, config):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.config = config

        self.crop_size = self.config["train_dataset"]["deepglobe"]["crop_size"]  # 256
        self.stride = self.config["train_dataset"]["deepglobe"]["stride"] # 128 for 50% overlap
        
        # Load model
        self.model_path = self.config['checkpoint_path']
        self.model = self.load_model(self.model_path)
        self.model.eval()

        # to avoid Deadloack  between CV Threads and Pytorch Threads caused in resizing
        cv2.setNumThreads(0)

        # preprocess
        self.angle_theta = self.config["train_dataset"]["angle_theta"]
        self.mean_bgr = np.array(eval(self.config["train_dataset"]["mean"]))
        self.deviation_bgr = np.array(eval(self.config["train_dataset"]["std"]))
        self.normalize_type = self.config["train_dataset"]["normalize_type"]

        # Load all test images path
        self.files = []
        for i in range(1, 51):  # Assuming 50 test images
            img_path = os.path.join(self.config["test_dir"], f"test_{i}", f"test_{i}.png")
            self.files.append(img_path)

    def reshape(self, image):

        if self.normalize_type == "Std":
            image = (image - self.mean_bgr) / (3 * self.deviation_bgr)
        elif self.normalize_type == "MinMax":
            image = (image - self.min_bgr) / (self.max_bgr - self.min_bgr)
            image = image * 2 - 1
        elif self.normalize_type == "Mean":
            image -= self.mean_bgr
        else:
            image = (image / 255.0) * 2 - 1

        image = image.transpose(2, 0, 1)
        return image

    def getRoadData(self, index):
        image_path = self.files[index]
        image = cv2.imread(image_path).astype(np.float32)
        image = self.reshape(image)
        image = torch.from_numpy(np.array(image))
        return image
    
    def load_model(self, model_path):
        """Load the pretrained model"""
        # Import your model architecture here
        self.multi_scale_pred = True

        model = StackHourglassNetMTL_DGCNv4(
        self.config["task1_classes"], self.config["task2_classes"]
        )
        model.cuda()
        
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint)
        model = model.to(self.device)
        return model
    
    def extract_patches(self, image, patch_size, stride):
        """Extract overlapping patches from image"""
        h, w = image.shape[1:]
        patches = []
        positions = []
        
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                # Adjust last patches to fit image boundaries
                y_end = min(y + patch_size, h)
                x_end = min(x + patch_size, w)
                y_start = y_end - patch_size
                x_start = x_end - patch_size
                
                patch = image[:, y_start:y_end, x_start:x_end]
                patches.append(patch)
                positions.append((y_start, x_start))
        
        return patches, positions
    
    def merge_predictions(self, predictions, positions, image_shape, patch_size):
        """Merge overlapping patch predictions using weighted averaging"""
        h, w = image_shape
        
        # For task 1 (segmentation)
        seg_sum = torch.zeros((self.config['task1_classes'], h, w), 
                              dtype=torch.float32, device=self.device)
        seg_count = torch.zeros((h, w), dtype=torch.float32, device=self.device)
        
        # Weight matrix for smooth blending (higher weight at center)
        weight = self.create_weight_matrix(patch_size)
        weight = weight.to(self.device)
        
        for pred_seg, (y, x) in zip(predictions, positions):
            y_end = y + patch_size
            x_end = x + patch_size
            
            # Apply weighted sum for segmentation
            seg_sum[:, y:y_end, x:x_end] += pred_seg * weight
            seg_count[y:y_end, x:x_end] += weight
        
        # Average predictions
        seg_count[seg_count == 0] = 1  # Avoid division by zero
        
        final_seg = seg_sum / seg_count.unsqueeze(0)
        
        return final_seg
    
    def create_weight_matrix(self, size):
        """Create Gaussian-like weight matrix for smooth blending"""
        center = size // 2
        y, x = np.ogrid[:size, :size]
        dist = np.sqrt((x - center)**2 + (y - center)**2)
        weight = np.exp(-(dist**2) / (2 * (size/4)**2))
        return torch.from_numpy(weight).float()
    
    @torch.no_grad()
    def predict_image(self, image):
        """Predict on a single test image"""
        c, h, w = image.shape
        
        # Extract patches
        patches, positions = self.extract_patches(
            image, self.crop_size, self.stride
        )
        
        # Predict on patches
        predictions = []
        for patch in patches:
            patch_batch = patch.unsqueeze(0).to(self.device)
            
            # Forward pass
            outputs, pred_vecmaps = self.model(patch_batch)
            
            # Extract task outputs (adjust based on your model output structure)
            if self.multi_scale_pred:
                outputs = outputs[-1]
                pred_vecmaps = pred_vecmaps[-1]
            
            # Apply softmax/sigmoid
            outputs = outputs[0, :, :, :]
            seg_pred = torch.sigmoid(outputs)  # [2, 256, 256]
            
            predictions.append(seg_pred)
        
        # Merge predictions
        final_seg = self.merge_predictions(
            predictions, positions, (h, w), self.crop_size
        )
        
        # Post-process
        seg_mask = (final_seg[1] > self.config['classification_threshold']).cpu().numpy()

        return seg_mask, final_seg
    
    def predict_test_set(self):
        """Predict on entire test set"""
        os.makedirs(self.config["output_dir"], exist_ok=True)
        
        for img_index in range(1, 51):
            print(f"Processing {img_index}...")
            image = self.getRoadData(img_index-1)
            seg_mask, final_seg = self.predict_image(image)
            
            # Save segmentation mask
            Image.fromarray((seg_mask * 255).astype(np.uint8)).save(
                os.path.join(self.config["output_dir"], f"{img_index}_prediction.png")
            )


def main(config):
    
    # Create inference engine
    engine = InferenceEngine(config)
    
    # Run inference
    engine.predict_test_set()
    
    print(f"Inference complete! Results saved to {config["output_dir"]}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=int, default=0)
    args = parser.parse_args()
    config_number = args.config
    config = configs[config_number]

    main(config)