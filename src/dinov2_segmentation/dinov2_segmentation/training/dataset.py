#!/usr/bin/env python3

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from transformers import AutoImageProcessor
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ADE20KDataset(Dataset):
    """
    ADE20K Dataset for semantic segmentation
    """
    def __init__(self, root_dir, split='training', transform=None, image_processor=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.image_processor = image_processor
        
        # ADE20K directory structure
        self.images_dir = os.path.join(root_dir, 'images', split)
        self.annotations_dir = os.path.join(root_dir, 'annotations', split)
        
        # Get all image files
        self.image_files = []
        if os.path.exists(self.images_dir):
            for file in os.listdir(self.images_dir):
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    self.image_files.append(file)
        
        self.image_files.sort()
        print(f"Found {len(self.image_files)} images in {split} split")
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        
        # For ADE20K, annotation files have the same name but .png extension
        mask_name = img_name.replace('.jpg', '.png').replace('.jpeg', '.png')
        mask_path = os.path.join(self.annotations_dir, mask_name)
        
        # Load image and mask
        image = Image.open(img_path).convert('RGB')
        
        if os.path.exists(mask_path):
            mask = Image.open(mask_path)
            # Convert to numpy and handle ADE20K format
            mask = np.array(mask)
            # ADE20K masks are 0-indexed, but we need to handle background (0) and objects (1-150)
            mask = mask.astype(np.int64)
        else:
            # Create dummy mask if annotation doesn't exist
            mask = np.zeros((image.size[1], image.size[0]), dtype=np.int64)
        
        # Apply transformations
        if self.transform:
            transformed = self.transform(image=np.array(image), mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # Use DINOv2 processor if provided
        if self.image_processor:
            # Convert back to PIL for processor
            if isinstance(image, torch.Tensor):
                image = image.permute(1, 2, 0).numpy().astype(np.uint8)
                image = Image.fromarray(image)
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image.astype(np.uint8))
            
            processed = self.image_processor(image, return_tensors="pt")
            pixel_values = processed["pixel_values"].squeeze(0)
            
            # Resize mask to match processed image if needed
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask).long()
        else:
            pixel_values = image
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask).long()
        
        return {
            'pixel_values': pixel_values,
            'labels': mask,
            'image_path': img_path
        }


def get_transforms(split='training', image_size=224):
    """
    Get data transforms for training/validation
    """
    if split == 'training':
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.ColorJitter(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])


def create_dataloaders(data_root, batch_size=8, num_workers=4, image_size=224):
    """
    Create training and validation dataloaders
    """
    # Initialize image processor
    image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    
    # Create transforms
    train_transform = get_transforms('training', image_size)
    val_transform = get_transforms('validation', image_size)
    
    # Create datasets
    train_dataset = ADE20KDataset(
        root_dir=data_root,
        split='training',
        transform=train_transform,
        image_processor=image_processor
    )
    
    val_dataset = ADE20KDataset(
        root_dir=data_root,
        split='validation',
        transform=val_transform,
        image_processor=image_processor
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


# ADE20K class names for reference
ADE20K_CLASSES = [
    'wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed', 'windowpane', 'grass',
    'cabinet', 'sidewalk', 'person', 'earth', 'door', 'table', 'mountain', 'plant', 'curtain', 'chair',
    'car', 'water', 'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug', 'field',
    'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe', 'lamp', 'bathtub', 'railing', 'cushion',
    'base', 'box', 'column', 'signboard', 'chest', 'counter', 'sand', 'sink', 'skyscraper', 'fireplace',
    'refrigerator', 'grandstand', 'path', 'stairs', 'runway', 'case', 'pool', 'pillow', 'screen', 'stairway',
    'river', 'bridge', 'bookcase', 'blind', 'coffee', 'toilet', 'flower', 'book', 'hill', 'bench',
    'countertop', 'stove', 'palm', 'kitchen', 'computer', 'swivel', 'boat', 'bar', 'arcade', 'hovel',
    'bus', 'towel', 'light', 'truck', 'tower', 'chandelier', 'awning', 'streetlight', 'booth', 'television',
    'airplane', 'dirt', 'apparel', 'pole', 'land', 'bannister', 'escalator', 'ottoman', 'bottle', 'buffet',
    'poster', 'stage', 'van', 'ship', 'fountain', 'conveyer', 'canopy', 'washer', 'plaything', 'swimming',
    'stool', 'barrel', 'basket', 'waterfall', 'tent', 'bag', 'minibike', 'cradle', 'oven', 'ball',
    'food', 'step', 'tank', 'trade', 'microwave', 'pot', 'animal', 'bicycle', 'lake', 'dishwasher',
    'screen', 'blanket', 'sculpture', 'hood', 'sconce', 'vase', 'traffic', 'tray', 'ashcan', 'fan',
    'pier', 'crt', 'plate', 'monitor', 'bulletin', 'shower', 'radiator', 'glass', 'clock', 'flag'
]