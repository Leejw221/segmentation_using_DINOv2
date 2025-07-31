#!/usr/bin/env python3

import torch
import numpy as np
from PIL import Image
import cv2
from transformers import AutoImageProcessor

from ..training.model import DINOv2ForSegmentation
from ..training.utils import create_color_palette, colorize_mask


class DINOv2SegmentationInference:
    """
    DINOv2 Segmentation Inference class
    """
    def __init__(self, model_path, num_classes=150, device=None):
        self.num_classes = num_classes
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = self._load_model(model_path)
        self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        
        # Create color palette for visualization
        self.color_palette = create_color_palette(num_classes)
        
        print(f"Inference model loaded on {self.device}")
        print(f"Number of classes: {num_classes}")
    
    def _load_model(self, model_path):
        """Load trained model"""
        model = DINOv2ForSegmentation(
            num_classes=self.num_classes,
            model_name="facebook/dinov2-base",
            freeze_backbone=True
        )
        
        # Load state dict
        state_dict = torch.load(model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in state_dict:
            model.load_state_dict(state_dict['model_state_dict'])
        else:
            model.load_state_dict(state_dict)
        
        model.to(self.device)
        model.eval()
        
        return model
    
    def preprocess_image(self, image):
        """
        Preprocess image for inference
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            preprocessed tensor
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        # Use DINOv2 processor
        inputs = self.processor(image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        
        return pixel_values, image
    
    def predict(self, image):
        """
        Perform segmentation prediction
        
        Args:
            image: Input image (PIL, numpy array, or file path)
            
        Returns:
            segmentation_map: numpy array of predicted classes
            confidence_map: numpy array of prediction confidence
        """
        pixel_values, original_image = self.preprocess_image(image)
        
        with torch.no_grad():
            outputs = self.model(pixel_values)
            
            # Get predictions and confidence
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            confidence = torch.max(probabilities, dim=1)[0]
            
            # Convert to numpy
            seg_map = predictions.squeeze(0).cpu().numpy()
            conf_map = confidence.squeeze(0).cpu().numpy()
        
        return seg_map, conf_map
    
    def visualize_segmentation(self, image, seg_map, alpha=0.6):
        """
        Create visualization of segmentation result
        
        Args:
            image: Original image
            seg_map: Segmentation map
            alpha: Transparency factor for overlay
            
        Returns:
            visualization: RGB image with segmentation overlay
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Resize segmentation map to match original image size
        orig_w, orig_h = image.size
        if seg_map.shape != (orig_h, orig_w):
            seg_map = cv2.resize(
                seg_map.astype(np.float32),
                (orig_w, orig_h),
                interpolation=cv2.INTER_NEAREST
            ).astype(np.uint8)
        
        # Convert to numpy
        image_np = np.array(image)
        
        # Create colored segmentation
        colored_seg = colorize_mask(seg_map, self.color_palette)
        
        # Blend images
        blended = cv2.addWeighted(
            image_np, 1 - alpha,
            colored_seg, alpha,
            0
        )
        
        return blended
    
    def predict_and_visualize(self, image, alpha=0.6):
        """
        Perform prediction and create visualization in one step
        
        Args:
            image: Input image
            alpha: Transparency for overlay
            
        Returns:
            seg_map: Segmentation map
            visualization: Blended visualization
            confidence: Confidence map
        """
        seg_map, confidence = self.predict(image)
        visualization = self.visualize_segmentation(image, seg_map, alpha)
        
        return seg_map, visualization, confidence
    
    def get_class_statistics(self, seg_map):
        """
        Get statistics about predicted classes
        
        Args:
            seg_map: Segmentation map
            
        Returns:
            dict with class statistics
        """
        unique_classes, counts = np.unique(seg_map, return_counts=True)
        total_pixels = seg_map.size
        
        stats = {}
        for class_id, count in zip(unique_classes, counts):
            percentage = (count / total_pixels) * 100
            stats[int(class_id)] = {
                'pixel_count': int(count),
                'percentage': float(percentage)
            }
        
        return stats


def main():
    """
    Simple test function
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='DINOv2 Segmentation Inference')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--image_path', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--output_path', type=str, default='output.png',
                       help='Path to save output')
    parser.add_argument('--num_classes', type=int, default=150,
                       help='Number of classes')
    
    args = parser.parse_args()
    
    # Create inference object
    inference = DINOv2SegmentationInference(
        model_path=args.model_path,
        num_classes=args.num_classes
    )
    
    # Perform inference
    seg_map, visualization, confidence = inference.predict_and_visualize(args.image_path)
    
    # Save result
    cv2.imwrite(args.output_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    print(f"Result saved to: {args.output_path}")
    
    # Print statistics
    stats = inference.get_class_statistics(seg_map)
    print("\nClass statistics:")
    for class_id, stat in stats.items():
        print(f"  Class {class_id}: {stat['pixel_count']} pixels ({stat['percentage']:.2f}%)")


if __name__ == '__main__':
    main()