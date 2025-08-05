#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoImageProcessor

class DINOv2ForSegmentation(nn.Module):
    """
    DINOv2 based segmentation model for ADE20K dataset
    Following DINOv2 paper implementation with simple BNHead
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
        
        # Simple segmentation head (DINOv2 paper style - BNHead)
        # BatchNorm + 1x1 Conv (same as DINOv2 paper)
        self.bn = nn.SyncBatchNorm(self.hidden_size)
        self.classifier = nn.Conv2d(self.hidden_size, num_classes, kernel_size=1)
        
        # Initialize classifier weights (following DINOv2 paper)
        nn.init.normal_(self.classifier.weight, std=0.01)
        nn.init.constant_(self.classifier.bias, 0)
        
        self.num_classes = num_classes
    
    def unfreeze_backbone(self):
        """Unfreeze the DINOv2 backbone for fine-tuning"""
        for param in self.dinov2.parameters():
            param.requires_grad = True
        print("DINOv2 backbone unfrozen for fine-tuning")
        
    def freeze_backbone(self):
        """Freeze the DINOv2 backbone"""
        for param in self.dinov2.parameters():
            param.requires_grad = False
        print("DINOv2 backbone frozen")
        
    def get_trainable_params_count(self):
        """Get count of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_head_parameters(self):
        """Get segmentation head parameters"""
        return list(self.bn.parameters()) + list(self.classifier.parameters())
        
    def forward(self, pixel_values):
        B, C, H, W = pixel_values.shape
        
        # Get DINOv2 features
        outputs = self.dinov2(pixel_values)
        features = outputs.last_hidden_state[:, 1:]  # Remove CLS token
        
        # Reshape to 2D feature map
        seq_len = features.shape[1]
        grid_size = int(seq_len ** 0.5)  # Should be H//patch_size, W//patch_size
        
        features = features.reshape(B, grid_size, grid_size, self.hidden_size)
        features = features.permute(0, 3, 1, 2)  # [B, hidden_size, grid_H, grid_W]
        
        # Apply simple segmentation head (DINOv2 paper style)
        features = self.bn(features)  # BatchNorm
        logits = self.classifier(features)  # 1x1 Conv
        
        # Upsample to input resolution (bilinear interpolation)
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
    Simple CrossEntropy loss for segmentation (DINOv2 paper style)
    """
    def __init__(self, ignore_index=0):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        
    def forward(self, pred, target):
        return self.ce_loss(pred, target)