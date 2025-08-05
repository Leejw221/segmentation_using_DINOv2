#!/usr/bin/env python3

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from transformers import AutoImageProcessor
from torchvision import transforms
import torchvision.transforms.functional as TF
import random

class ADE20KDataset(Dataset):
    """
    ADE20K Dataset for semantic segmentation - OpenCV free version
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
            
            # 🔥 ADE20K 특수 처리: 값을 0-149 범위로 제한
            # ADE20K는 1-150으로 인덱스되어 있지만, 우리는 0-149 사용
            mask = mask - 1  # 1-150 → 0-149로 변환
            mask = np.clip(mask, 0, 149)  # 범위 제한
            mask = mask.astype(np.uint8)
            
            # 디버그: 처음 몇 개 이미지만 범위 확인
            if idx < 5:
                print(f"Image {idx}: Mask min: {mask.min()}, max: {mask.max()}, unique classes: {len(np.unique(mask))}")
        else:
            # Create dummy mask if annotation doesn't exist
            mask = np.zeros((image.size[1], image.size[0]), dtype=np.uint8)
        
        # Apply transformations
        if self.transform:
            image, mask = self.transform(image, mask)
        
        # Use DINOv2 processor if provided
        if self.image_processor:
            # Convert back to PIL for processor if needed
            if isinstance(image, torch.Tensor):
                image = TF.to_pil_image(image)
            
            processed = self.image_processor(image, return_tensors="pt")
            pixel_values = processed["pixel_values"].squeeze(0)
            
            # Convert mask to tensor if needed
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask.copy()).long()
        else:
            # Convert image to tensor if needed
            if isinstance(image, Image.Image):
                image = TF.to_tensor(image)
            pixel_values = image
            
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask.copy()).long()
        
        return {
            'pixel_values': pixel_values,
            'labels': mask,
            'image_path': img_path
        }


class SegmentationTransform:
    """
    DINOv2 paper style transforms for semantic segmentation
    """
    def __init__(self, image_size=640, is_training=True, scale_range=(1.0, 3.0)):
        self.image_size = image_size
        self.is_training = is_training
        self.scale_range = scale_range
        
    def __call__(self, image, mask):
        if self.is_training:
            # Multi-scale crop (DINOv2 paper style)
            scale = random.uniform(self.scale_range[0], self.scale_range[1])
            scaled_size = int(self.image_size * scale)
            
            # Random crop from scaled image
            image = TF.resize(image, (scaled_size, scaled_size), interpolation=Image.BILINEAR)
            mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
            mask_tensor = TF.resize(mask_tensor, (scaled_size, scaled_size), interpolation=Image.NEAREST)
            mask = mask_tensor.squeeze(0).numpy().astype(np.uint8)
            
            # Random crop to target size
            if scaled_size > self.image_size:
                i = random.randint(0, scaled_size - self.image_size)
                j = random.randint(0, scaled_size - self.image_size)
                image = TF.crop(image, i, j, self.image_size, self.image_size)
                mask = mask[i:i+self.image_size, j:j+self.image_size]
            else:
                # Resize to target size if scale < 1.0
                image = TF.resize(image, (self.image_size, self.image_size), interpolation=Image.BILINEAR)
                mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
                mask_tensor = TF.resize(mask_tensor, (self.image_size, self.image_size), interpolation=Image.NEAREST)
                mask = mask_tensor.squeeze(0).numpy().astype(np.uint8)
            
            # Random horizontal flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = np.fliplr(mask).copy()
            
            # Photometric distortion (DINOv2 paper style)
            if random.random() > 0.5:
                brightness_factor = random.uniform(0.8, 1.2)
                image = TF.adjust_brightness(image, brightness_factor)
            
            if random.random() > 0.5:
                contrast_factor = random.uniform(0.8, 1.2)
                image = TF.adjust_contrast(image, contrast_factor)
                
            if random.random() > 0.5:
                saturation_factor = random.uniform(0.8, 1.2)
                image = TF.adjust_saturation(image, saturation_factor)
                
            if random.random() > 0.5:
                hue_factor = random.uniform(-0.1, 0.1)
                image = TF.adjust_hue(image, hue_factor)
        else:
            # Validation: simple resize
            image = TF.resize(image, (self.image_size, self.image_size), interpolation=Image.BILINEAR)
            mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
            mask_tensor = TF.resize(mask_tensor, (self.image_size, self.image_size), interpolation=Image.NEAREST)
            mask = mask_tensor.squeeze(0).numpy().astype(np.uint8)
        
        # Convert to tensor and normalize
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        return image, mask


def get_transforms(split='training', image_size=640):
    """
    Get data transforms for training/validation (DINOv2 paper style)
    """
    if split == 'training':
        return SegmentationTransform(image_size=image_size, is_training=True, scale_range=(1.0, 3.0))
    else:
        return SegmentationTransform(image_size=image_size, is_training=False)


def create_dataloaders(data_root, batch_size=4, num_workers=6, image_size=640, distributed=False):
    """
    Create training and validation dataloaders (DINOv2 paper style)
    """
    # Initialize image processor (not used in DINOv2 style, we handle transforms manually)
    # image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    
    # Create transforms (DINOv2 style)
    train_transform = get_transforms('training', image_size)
    val_transform = get_transforms('validation', image_size)
    
    # Create datasets
    train_dataset = ADE20KDataset(
        root_dir=data_root,
        split='training',
        transform=train_transform,
        image_processor=None  # We handle transforms manually
    )
    
    val_dataset = ADE20KDataset(
        root_dir=data_root,
        split='validation',
        transform=val_transform,
        image_processor=None  # We handle transforms manually
    )
    
    # Create samplers for distributed training
    train_sampler = None
    val_sampler = None
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=False,  # Disabled for memory safety
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=False  # Disabled for memory safety
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