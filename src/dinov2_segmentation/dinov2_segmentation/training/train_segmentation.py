#!/usr/bin/env python3

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
import yaml
from tqdm import tqdm
import numpy as np
from datetime import datetime

from .model import DINOv2ForSegmentation, SegmentationLoss
from .dataset import create_dataloaders
from .utils import calculate_miou, save_checkpoint, load_checkpoint


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_samples = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(pixel_values)
        
        # Calculate loss
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item() * pixel_values.size(0)
        total_samples += pixel_values.size(0)
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Avg Loss': f'{total_loss / total_samples:.4f}'
        })
    
    return total_loss / total_samples


def validate_epoch(model, val_loader, criterion, device, num_classes=150):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for batch in pbar:
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(pixel_values)
            loss = criterion(outputs, labels)
            
            # Update loss
            total_loss += loss.item() * pixel_values.size(0)
            total_samples += pixel_values.size(0)
            
            # Get predictions
            preds = torch.argmax(outputs, dim=1)
            
            # Store for mIoU calculation
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    # Calculate mIoU
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    miou = calculate_miou(all_preds, all_labels, num_classes)
    
    return total_loss / total_samples, miou


def main():
    parser = argparse.ArgumentParser(description='Train DINOv2 Segmentation Model')
    parser.add_argument('--config', type=str, default='configs/config.yaml', 
                       help='Path to config file')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Path to ADE20K dataset root')
    parser.add_argument('--output_dir', type=str, default='models/checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Load config
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Default config
        config = {
            'model': {
                'num_classes': 150,
                'model_name': 'facebook/dinov2-base',
                'freeze_backbone': True
            },
            'training': {
                'batch_size': 8,
                'learning_rate': 1e-4,
                'num_epochs': 50,
                'image_size': 224,
                'num_workers': 4
            },
            'loss': {
                'use_dice': False,
                'dice_weight': 1.0
            }
        }
    
    # Ensure numeric values are properly typed
    config['training']['learning_rate'] = float(config['training']['learning_rate'])
    config['training']['batch_size'] = int(config['training']['batch_size'])
    config['training']['num_epochs'] = int(config['training']['num_epochs'])
    config['training']['image_size'] = int(config['training']['image_size'])
    config['training']['num_workers'] = int(config['training']['num_workers'])
    config['model']['num_classes'] = int(config['model']['num_classes'])
    config['loss']['dice_weight'] = float(config['loss']['dice_weight'])
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join('logs', f'dinov2_seg_{timestamp}')
    writer = SummaryWriter(log_dir)
    
    # Create model
    model = DINOv2ForSegmentation(
        num_classes=config['model']['num_classes'],
        model_name=config['model']['model_name'],
        freeze_backbone=config['model']['freeze_backbone']
    ).to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")
    
    # Create data loaders
    train_loader, val_loader = create_dataloaders(
        data_root=args.data_root,
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        image_size=config['training']['image_size']
    )
    
    # Setup loss and optimizer
    criterion = SegmentationLoss(
        use_dice=config['loss']['use_dice'],
        dice_weight=config['loss']['dice_weight']
    )
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(config['training']['learning_rate']),
        weight_decay=1e-4
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['training']['num_epochs']
    )
    
    # Resume training if specified
    start_epoch = 0
    best_miou = 0.0
    
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_miou = checkpoint.get('best_miou', 0.0)
    
    # Training loop
    print("Starting training...")
    for epoch in range(start_epoch, config['training']['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['training']['num_epochs']}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch + 1)
        
        # Validate
        val_loss, val_miou = validate_epoch(
            model, val_loader, criterion, device, 
            config['model']['num_classes']
        )
        
        # Update scheduler
        scheduler.step()
        
        # Log metrics
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('mIoU/Validation', val_miou, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val mIoU: {val_miou:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save checkpoint
        is_best = val_miou > best_miou
        if is_best:
            best_miou = val_miou
        
        checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch + 1}.pth')
        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_miou': val_miou,
            'best_miou': best_miou,
            'config': config
        }, checkpoint_path, is_best)
        
        # Save best model separately
        if is_best:
            best_model_path = os.path.join(args.output_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with mIoU: {best_miou:.4f}")
    
    writer.close()
    print(f"Training completed! Best mIoU: {best_miou:.4f}")


if __name__ == '__main__':
    main()