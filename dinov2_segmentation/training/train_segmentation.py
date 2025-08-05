#!/usr/bin/env python3

import os
import sys
import time
import argparse
import yaml
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from dinov2_segmentation.training.model import DINOv2ForSegmentation, SegmentationLoss
from dinov2_segmentation.training.dataset import create_dataloaders
from dinov2_segmentation.training.utils import AverageMeter, intersectionAndUnionGPU, calculate_miou


def get_cosine_lr_scheduler(optimizer, max_iters, min_lr=0.0):
    """Cosine learning rate scheduler"""
    def cosine_lr_lambda(current_iter):
        if current_iter < 0:
            return 1.0
        else:
            return min_lr + (1 - min_lr) * 0.5 * (1 + np.cos(np.pi * current_iter / max_iters))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_lr_lambda)


def get_linear_warmup_scheduler(optimizer, warmup_iters, warmup_factor=1e-6):
    """Linear warmup scheduler"""
    def warmup_lr_lambda(current_iter):
        if current_iter < warmup_iters:
            alpha = current_iter / warmup_iters
            return warmup_factor + (1.0 - warmup_factor) * alpha
        else:
            return 1.0
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lr_lambda)


def create_optimizer(model, config, phase):
    """Create optimizer for current phase"""
    if phase == 1:
        # Phase 1: Only head parameters (backbone frozen)
        params = model.get_head_parameters()
        lr = config['training'][f'phase1_learning_rate']
        weight_decay = config['training'][f'phase1_weight_decay']
        betas = config['training'][f'phase1_betas']
    else:
        # Phase 2: All parameters (backbone unfrozen)
        params = model.parameters()
        lr = config['training'][f'phase2_learning_rate']
        weight_decay = config['training'][f'phase2_weight_decay']
        betas = config['training'][f'phase2_betas']
    
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas)
    return optimizer


def validate(model, val_loader, criterion, device, num_classes=150):
    """Validation function"""
    model.eval()
    losses = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            images = batch['pixel_values'].to(device)
            targets = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            # Update loss
            losses.update(loss.item(), images.size(0))
            
            # Calculate IoU
            outputs = torch.argmax(outputs, dim=1)
            intersection, union, target = intersectionAndUnionGPU(
                outputs, targets, num_classes, ignore_index=0
            )
            
            intersection_meter.update(intersection.cpu().numpy())
            union_meter.update(union.cpu().numpy())
            target_meter.update(target.cpu().numpy())
    
    # Calculate mIoU
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    
    return losses.avg, mIoU, mAcc, allAcc


def train_phase(model, train_loader, val_loader, config, phase, device, output_dir):
    """Train one phase - Single GPU version"""
    
    # Phase configurations
    phase_iters = config['training'][f'phase{phase}_iterations']
    warmup_iters = config['training']['warmup_iterations'] if phase == 1 else 0
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config, phase)
    
    if phase == 1:
        # Phase 1: Linear warmup + Cosine decay
        warmup_scheduler = get_linear_warmup_scheduler(
            optimizer, warmup_iters, config['training']['warmup_factor']
        )
        main_scheduler = get_cosine_lr_scheduler(
            optimizer, phase_iters, config['training']['min_lr']
        )
    else:
        # Phase 2: Cosine decay only
        main_scheduler = get_cosine_lr_scheduler(
            optimizer, phase_iters, config['training']['min_lr']
        )
        warmup_scheduler = None
    
    # Loss function
    criterion = SegmentationLoss(ignore_index=config['loss']['ignore_index'])
    
    # TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'logs', f'phase{phase}'))
    
    # Training loop
    model.train()
    losses = AverageMeter()
    best_miou = 0.0
    
    # 메트릭 추적을 위한 변수들
    current_train_loss = 0.0
    current_val_loss = 0.0
    current_val_miou = 0.0
    current_val_train_ratio = 0.0
    
    print(f"\n{'='*60}")
    print(f"Starting Phase {phase} Training")
    print(f"{'='*60}")
    print(f"Iterations: {phase_iters}")
    print(f"Warmup iterations: {warmup_iters}")
    print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
    print(f"Trainable parameters: {model.get_trainable_params_count():,}")
    
    for iteration in range(phase_iters):
        for batch in train_loader:
            images = batch['pixel_values'].to(device)
            targets = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update learning rate
            if warmup_scheduler and iteration < warmup_iters:
                warmup_scheduler.step()
            else:
                main_scheduler.step()
            
            # Update metrics
            losses.update(loss.item(), images.size(0))
            current_train_loss = losses.avg
            
            # Logging
            if iteration % config['logging']['print_interval'] == 0:
                lr = optimizer.param_groups[0]['lr']
                print(f"Phase {phase} Iter {iteration:5d}/{phase_iters}: "
                      f"Loss: {losses.avg:.4f}, LR: {lr:.2e}")
                
                writer.add_scalar('train/loss', losses.avg, iteration)
                writer.add_scalar('train/lr', lr, iteration)
            
            # Validation
            if iteration % config['evaluation']['interval'] == 0 and iteration > 0:
                val_loss, val_miou, val_macc, val_allacc = validate(
                    model, val_loader, criterion, device, config['model']['num_classes']
                )
                
                # 메트릭 업데이트
                current_val_loss = val_loss
                current_val_miou = val_miou
                current_val_train_ratio = val_loss / (current_train_loss + 1e-8)
                
                print(f"Validation - Loss: {val_loss:.4f}, mIoU: {val_miou:.4f}, "
                      f"mAcc: {val_macc:.4f}, allAcc: {val_allacc:.4f}")
                print(f"Val/Train Loss Ratio: {current_val_train_ratio:.3f}")
                
                writer.add_scalar('val/loss', val_loss, iteration)
                writer.add_scalar('val/mIoU', val_miou, iteration)
                writer.add_scalar('val/mAcc', val_macc, iteration)
                writer.add_scalar('val/allAcc', val_allacc, iteration)
                writer.add_scalar('metrics/val_train_ratio', current_val_train_ratio, iteration)
                
                # Save best model
                if val_miou > best_miou:
                    best_miou = val_miou
                    torch.save({
                        'iteration': iteration,
                        'phase': phase,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_miou': val_miou,
                        'val_loss': val_loss,
                        'train_loss': current_train_loss,
                        'val_train_ratio': current_val_train_ratio,
                        'best_miou': best_miou,
                        'config': config
                    }, os.path.join(output_dir, f'best_model_phase{phase}.pth'))
                    
                    print(f"New best mIoU: {best_miou:.4f} - Model saved!")
                
                model.train()
            
            break  # Process only one batch per iteration
    
    # Save final model
    torch.save({
        'iteration': phase_iters,
        'phase': phase,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_miou': current_val_miou,
        'val_loss': current_val_loss,
        'train_loss': current_train_loss,
        'val_train_ratio': current_val_train_ratio,
        'best_miou': best_miou,
        'config': config
    }, os.path.join(output_dir, f'final_model_phase{phase}.pth'))
    
    writer.close()
    print(f"Phase {phase} completed!")
    print(f"Final mIoU: {current_val_miou:.4f} | Best mIoU: {best_miou:.4f}")
    print(f"Final Train Loss: {current_train_loss:.4f} | Val Loss: {current_val_loss:.4f}")
    print(f"Final Val/Train Ratio: {current_val_train_ratio:.3f}")
    
    return best_miou


def main():
    """Main function - Single GPU version"""
    parser = argparse.ArgumentParser(description='DINOv2 Segmentation Training - Single GPU')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update paths - use current working directory as base
    if not os.path.isabs(config['data']['ade20k_root']):
        current_dir = os.getcwd()
        config['data']['ade20k_root'] = os.path.join(current_dir, config['data']['ade20k_root'])
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Setup device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)
    
    # Create dataloaders (Non-distributed)
    train_loader, val_loader = create_dataloaders(
        data_root=config['data']['ade20k_root'],
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        image_size=config['training']['image_size'],
        distributed=False  # Single GPU
    )
    
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Val dataset size: {len(val_loader.dataset)}")
    
    # Create model
    model = DINOv2ForSegmentation(
        num_classes=config['model']['num_classes'],
        model_name=config['model']['model_name'],
        freeze_backbone=True  # Start with frozen backbone
    ).to(device)
    
    print(f"Model created with {model.get_trainable_params_count():,} trainable parameters")
    
    # Phase 1: Linear Probing (Backbone Frozen)
    print("\n🔥 Phase 1: Linear Probing (Backbone Frozen)")
    
    phase1_miou = train_phase(
        model, train_loader, val_loader, config, phase=1, 
        device=device, output_dir=args.output_dir
    )
    
    # Phase 2: Fine-tuning (Backbone Unfrozen)
    print("\n🚀 Phase 2: Fine-tuning (Backbone Unfrozen)")
    
    # Unfreeze backbone for Phase 2
    model.unfreeze_backbone()
    
    phase2_miou = train_phase(
        model, train_loader, val_loader, config, phase=2,
        device=device, output_dir=args.output_dir
    )
    
    # Final results
    print(f"\n{'='*60}")
    print("Training Completed!")
    print(f"{'='*60}")
    print(f"Phase 1 Best mIoU: {phase1_miou:.4f}")
    print(f"Phase 2 Best mIoU: {phase2_miou:.4f}")
    print(f"Improvement: {phase2_miou - phase1_miou:.4f}")
    print(f"Output directory: {args.output_dir}")


if __name__ == '__main__':
    main()