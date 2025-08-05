#!/usr/bin/env python3

import torch
import numpy as np
import os
from sklearn.metrics import confusion_matrix
from typing import Tuple


def intersectionAndUnionGPU(output: torch.Tensor, target: torch.Tensor, 
                           K: int, ignore_index: int = 255) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate intersection and union for semantic segmentation on GPU
    
    Args:
        output: predicted segmentation map [N, H, W]
        target: ground truth segmentation map [N, H, W] 
        K: number of classes
        ignore_index: index to ignore in calculation
        
    Returns:
        intersection: intersection for each class [K]
        union: union for each class [K]  
        target: target count for each class [K]
    """
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape
    
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    
    intersection = output[output == target]
    area_intersection = torch.histc(intersection.float(), bins=K, min=0, max=K-1)
    area_output = torch.histc(output.float(), bins=K, min=0, max=K-1)
    area_target = torch.histc(target.float(), bins=K, min=0, max=K-1)
    area_union = area_output + area_target - area_intersection
    
    return area_intersection, area_union, area_target


def calculate_miou(predictions, labels, num_classes, ignore_index=255):
    """
    Calculate mean Intersection over Union (mIoU)
    """
    # Flatten arrays
    predictions = predictions.flatten()
    labels = labels.flatten()
    
    # Remove ignore index
    valid_mask = labels != ignore_index
    predictions = predictions[valid_mask]
    labels = labels[valid_mask]
    
    # Calculate confusion matrix
    cm = confusion_matrix(labels, predictions, labels=range(num_classes))
    
    # Calculate IoU for each class
    intersection = np.diag(cm)
    union = cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm)
    
    # Avoid division by zero
    iou = intersection / np.maximum(union, 1)
    
    # Calculate mean IoU (excluding classes not present in ground truth)
    valid_classes = union > 0
    miou = np.mean(iou[valid_classes])
    
    return miou


def calculate_pixel_accuracy(predictions, labels, ignore_index=255):
    """
    Calculate pixel accuracy
    """
    predictions = predictions.flatten()
    labels = labels.flatten()
    
    # Remove ignore index
    valid_mask = labels != ignore_index
    predictions = predictions[valid_mask]
    labels = labels[valid_mask]
    
    correct = (predictions == labels).sum()
    total = len(labels)
    
    return correct / total if total > 0 else 0.0


def save_checkpoint(state, filepath, is_best=False):
    """
    Save model checkpoint
    """
    torch.save(state, filepath)
    print(f"Checkpoint saved: {filepath}")
    
    if is_best:
        best_filepath = filepath.replace('.pth', '_best.pth')
        torch.save(state, best_filepath)
        print(f"Best checkpoint saved: {best_filepath}")


def load_checkpoint(filepath):
    """
    Load model checkpoint
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    checkpoint = torch.load(filepath, map_location='cpu')
    print(f"Checkpoint loaded: {filepath}")
    
    return checkpoint


def get_lr(optimizer):
    """
    Get current learning rate from optimizer
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


class AverageMeter:
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def create_color_palette(num_classes=150):
    """
    Create color palette for visualization
    """
    np.random.seed(42)  # For consistent colors
    palette = np.random.randint(0, 255, (num_classes, 3), dtype=np.uint8)
    palette[0] = [0, 0, 0]  # Background as black
    return palette


def colorize_mask(mask, palette):
    """
    Convert segmentation mask to colored image
    """
    colored_mask = palette[mask]
    return colored_mask


def print_model_summary(model):
    """
    Print model parameter summary
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model Summary:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters: {total_params - trainable_params:,}")
    print(f"  Trainable ratio: {trainable_params / total_params * 100:.1f}%")


def setup_directories(base_dir):
    """
    Setup directory structure for training
    """
    dirs = {
        'checkpoints': os.path.join(base_dir, 'checkpoints'),
        'logs': os.path.join(base_dir, 'logs'),
        'outputs': os.path.join(base_dir, 'outputs'),
        'visualizations': os.path.join(base_dir, 'visualizations')
    }
    
    for name, path in dirs.items():
        os.makedirs(path, exist_ok=True)
        print(f"Created directory: {path}")
    
    return dirs