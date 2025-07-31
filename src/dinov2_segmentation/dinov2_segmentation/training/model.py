#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoImageProcessor

class DINOv2ForSegmentation(nn.Module):
    """
    DINOv2 based segmentation model for ADE20K dataset
    Inspired by RISE-2 implementation
    """
    def __init__(self, num_classes=150, model_name="facebook/dinov2-base", freeze_backbone=True):
        super().__init__()
        
        # Load DINOv2 backbone
        self.dinov2 = AutoModel.from_pretrained(model_name)
        self.patch_size = self.dinov2.config.patch_size  # Usually 14
        self.hidden_size = self.dinov2.config.hidden_size  # 768 for base
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.dinov2.parameters():
                param.requires_grad = False
        
        # Segmentation head - similar to RISE-2 approach
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(self.hidden_size, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
        
        self.num_classes = num_classes
        
    def forward(self, pixel_values):
        B, C, H, W = pixel_values.shape
        
        # Get DINOv2 features
        outputs = self.dinov2(pixel_values)
        features = outputs.last_hidden_state[:, 1:]  # Remove CLS token
        
        # Reshape to 2D feature map (following RISE-2 approach)
        seq_len = features.shape[1]
        grid_size = int(seq_len ** 0.5)  # Should be H//patch_size, W//patch_size
        
        features = features.reshape(B, grid_size, grid_size, self.hidden_size)
        features = features.permute(0, 3, 1, 2)  # [B, hidden_size, grid_H, grid_W]
        
        # Apply segmentation head
        logits = self.segmentation_head(features)
        
        # Upsample to input resolution
        logits = F.interpolate(
            logits, 
            size=(H, W), 
            mode='bilinear', 
            align_corners=False
        )
        
        return logits
    
    def get_feature_extractor(self):
        """Get the image processor for preprocessing"""
        return AutoImageProcessor.from_pretrained("facebook/dinov2-base")


class SegmentationLoss(nn.Module):
    """
    Segmentation loss combining CrossEntropy and optional Dice loss
    """
    def __init__(self, ignore_index=255, use_dice=False, dice_weight=1.0):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.use_dice = use_dice
        self.dice_weight = dice_weight
        
    def dice_loss(self, pred, target, smooth=1e-6):
        """Compute Dice loss"""
        pred = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
        
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        
        dice = (2 * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()
        
    def forward(self, pred, target):
        ce_loss = self.ce_loss(pred, target)
        
        if self.use_dice:
            dice_loss = self.dice_loss(pred, target)
            return ce_loss + self.dice_weight * dice_loss
        
        return ce_loss