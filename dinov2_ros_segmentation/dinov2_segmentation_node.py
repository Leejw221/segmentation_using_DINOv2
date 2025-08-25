#!/usr/bin/env python3

import os
import math
import itertools
import urllib.request

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image, PointCloud2, PointField, CameraInfo
from std_msgs.msg import Header
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
import cv2
import numpy as np
from PIL import Image as PILImage

import torch
import torch.nn as nn
import torch.nn.functional as F

# DINOv2 will be loaded via torch.hub (no local path needed)
# Removed local dinov2 path dependency for cleaner installation

# Multiscale img_ratios presets
FAST = [1.0, 1.25]
# [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
# [1.0, 1.32]

# ADE20K class names (1-indexed, 150 classes total)
ADE20K_CLASSES = [
    'background', 'wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed',
    'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth', 'door', 'table',
    'mountain', 'plant', 'curtain', 'chair', 'car', 'water', 'painting', 'sofa',
    'shelf', 'house', 'sea', 'mirror', 'rug', 'field', 'armchair', 'seat', 'fence',
    'desk', 'rock', 'wardrobe', 'lamp', 'bathtub', 'railing', 'cushion', 'base',
    'box', 'column', 'signboard', 'chest', 'counter', 'sand', 'sink', 'skyscraper',
    'fireplace', 'refrigerator', 'grandstand', 'path', 'stairs', 'runway', 'case',
    'pool', 'pillow', 'screen', 'stairway', 'river', 'bridge', 'bookcase', 'blind',
    'coffee', 'toilet', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove',
    'palm', 'kitchen', 'computer', 'swivel', 'boat', 'bar', 'arcade', 'hovel',
    'bus', 'towel', 'light', 'truck', 'tower', 'chandelier', 'awning', 'streetlight',
    'booth', 'television', 'airplane', 'dirt', 'apparel', 'pole', 'land', 'bannister',
    'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van', 'ship',
    'fountain', 'conveyer', 'canopy', 'washer', 'plaything', 'swimming', 'stool',
    'barrel', 'basket', 'waterfall', 'tent', 'bag', 'minibike', 'cradle', 'oven',
    'ball', 'food', 'step', 'tank', 'trade', 'microwave', 'pot', 'animal', 'bicycle',
    'lake', 'dishwasher', 'screen', 'blanket', 'sculpture', 'hood', 'sconce', 'vase',
    'traffic', 'tray', 'ashcan', 'fan', 'pier', 'crt', 'plate', 'monitor', 'bulletin',
    'shower', 'radiator', 'glass', 'clock', 'flag'
]

# ADE20K Official Colormap - 150 classes + background (151 total)
# Index 0: background, Index 1-150: object classes
# Lab objects: chair(20)=빨간색, box(42)=주황색, bottle(99)=연두색, ball(120)=자홍색, monitor(144)=주황색
ADE20K_COLORMAP = [
    (0, 0, 0), (120, 120, 120), (180, 120, 120), (6, 230, 230), (80, 50, 50),
    (4, 200, 3), (120, 120, 80), (140, 140, 140), (204, 5, 255), (230, 230, 230),
    (4, 250, 7), (224, 5, 255), (235, 255, 7), (150, 5, 61), (120, 120, 70),
    (8, 255, 51), (255, 6, 82), (143, 255, 140), (204, 255, 4), (255, 51, 7),
    (204, 70, 3), (0, 102, 200), (61, 230, 250), (255, 6, 51), (11, 102, 255),
    (255, 7, 71), (255, 9, 224), (9, 7, 230), (220, 220, 220), (255, 9, 92),
    (112, 9, 255), (8, 255, 214), (7, 255, 224), (255, 184, 6), (10, 255, 71),
    (255, 41, 10), (7, 255, 255), (224, 255, 8), (102, 8, 255), (255, 61, 6),
    (255, 194, 7), (255, 122, 8), (0, 255, 20), (255, 8, 41), (255, 5, 153),
    (6, 51, 255), (235, 12, 255), (160, 150, 20), (0, 163, 255), (140, 140, 140),
    (250, 10, 15), (20, 255, 0), (31, 255, 0), (255, 31, 0), (255, 224, 0),
    (153, 255, 0), (0, 0, 255), (255, 71, 0), (0, 235, 255), (0, 173, 255),
    (31, 0, 255), (11, 200, 200), (255, 82, 0), (0, 255, 245), (0, 61, 255),
    (0, 255, 112), (0, 255, 133), (255, 0, 0), (255, 163, 0), (255, 102, 0),
    (194, 255, 0), (0, 143, 255), (51, 255, 0), (0, 82, 255), (0, 255, 41),
    (0, 255, 173), (10, 0, 255), (173, 255, 0), (0, 255, 153), (255, 92, 0),
    (255, 0, 255), (255, 0, 245), (255, 0, 102), (255, 173, 0), (255, 0, 20),
    (255, 184, 184), (0, 31, 255), (0, 255, 61), (0, 71, 255), (255, 0, 204),
    (0, 255, 194), (0, 255, 82), (0, 10, 255), (0, 112, 255), (51, 0, 255),
    (0, 194, 255), (0, 122, 255), (0, 255, 163), (255, 153, 0), (0, 255, 10),
    (255, 112, 0), (143, 255, 0), (82, 0, 255), (163, 255, 0), (255, 235, 0),
    (8, 184, 170), (133, 0, 255), (0, 255, 92), (184, 0, 255), (255, 0, 31),
    (0, 184, 255), (0, 214, 255), (255, 0, 112), (92, 255, 0), (0, 224, 255),
    (112, 224, 255), (70, 184, 160), (163, 0, 255), (153, 0, 255), (71, 255, 0),
    (255, 0, 163), (255, 204, 0), (255, 0, 143), (0, 255, 235), (133, 255, 0),
    (255, 0, 235), (245, 0, 255), (255, 0, 122), (255, 245, 0), (10, 190, 212),
    (214, 255, 0), (0, 204, 255), (20, 0, 255), (255, 255, 0), (0, 153, 255),
    (0, 41, 255), (0, 255, 204), (41, 0, 255), (41, 255, 0), (173, 0, 255),
    (0, 245, 255), (71, 0, 255), (122, 0, 255), (0, 255, 184), (0, 92, 255),
    (184, 255, 0), (0, 133, 255), (255, 214, 0), (25, 194, 194), (102, 255, 0),
    (92, 0, 255),
]

# Extend colormap to 151 classes (ADE20K has 150 classes + background)
while len(ADE20K_COLORMAP) < 151:
    ADE20K_COLORMAP.append((128, 128, 128))

# Lab objects class IDs (1-indexed in ADE20K) - Only target objects
# Color format: (R, G, B) from ADE20K_COLORMAP[class_id + 1]
LAB_OBJECT_CLASSES = {
    20: "chair",           # 의자   - Color: (255, 51, 7)   = 선명한 빨간색
    28: "mirror",          # 거울   - Color: (220, 220, 220) = 회색
    42: "box",             # 상자   - Color: (255, 8, 41)   = 진한 빨간색  
    68: "book",            # 책     - Color: (0, 255, 133)  = 청록색
    99: "bottle",          # 병     - Color: (153, 255, 0)  = 밝은 연두색
    120: "ball",           # 공     - Color: (255, 0, 245)  = 밝은 자홍색
    144: "monitor",        # 모니터 - Color: (255, 153, 0)  = 밝은 주황색
}


class CenterPadding(torch.nn.Module):
    def __init__(self, multiple):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    @torch.inference_mode()
    def forward(self, x):
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        output = F.pad(x, pads)
        return output


class LinearSegmentationHead(nn.Module):
    """Simple linear segmentation head for single-scale inference"""
    def __init__(self, in_channels, num_classes=150):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv_seg = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        
    def forward(self, x):
        # x is single feature tensor from backbone
        if isinstance(x, (list, tuple)):
            x = x[0]  # Take first (and only) feature
        x = self.bn(x)
        x = self.conv_seg(x)
        return x


class BNHead(nn.Module):
    """
    Official DINOv2 BNHead implementation matching pretrained weights
    Based on resize_concat input transform with SyncBN
    """
    def __init__(self, num_classes=150, backbone_size='base'):
        super().__init__()
        # Official hardcoded values from DINOv2 BNHead
        hardcoded_channels = {
            'small': 384 * 4,   # 1536
            'base': 768 * 4,    # 3072  
            'large': 1024 * 4,  # 4096
            'giant': 1536 * 4   # 6144
        }
        
        # Use official hardcoded value instead of calculated one
        self.in_channels = hardcoded_channels.get(backbone_size, 768 * 4)
        
        # Use SyncBatchNorm for better performance (falls back to BatchNorm if needed)
        try:
            self.bn = nn.SyncBatchNorm(self.in_channels)
        except:
            self.bn = nn.BatchNorm2d(self.in_channels)
            
        self.conv_seg = nn.Conv2d(self.in_channels, num_classes, kernel_size=1)
        self.align_corners = False
        
    def _resize_concat_transform(self, inputs):
        """
        Official MMSeg _transform_inputs logic for resize_concat
        Based on DINOv2 BNHead implementation
        """
        if not isinstance(inputs, (list, tuple)):
            return inputs
        
        # Handle nested lists (from MMSeg logic)
        input_list = []
        for x in inputs:
            if isinstance(x, list):
                input_list.extend(x)
            else:
                input_list.append(x)
        inputs = input_list
        
        # Convert 2D tensors to 4D (from MMSeg: image descriptors can be 1x1 resolution)
        for i, x in enumerate(inputs):
            if len(x.shape) == 2:
                inputs[i] = x[:, :, None, None]
        
        # Use the first feature's spatial size as target
        if len(inputs) == 0:
            return None
            
        target_h, target_w = inputs[0].shape[-2:]
        
        # Resize all features to the same spatial size
        upsampled_inputs = []
        for x in inputs:
            if x.shape[-2:] != (target_h, target_w):
                x = F.interpolate(
                    x,
                    size=(target_h, target_w),
                    mode='bilinear',
                    align_corners=self.align_corners
                )
            upsampled_inputs.append(x)
        
        # Concatenate along channel dimension
        return torch.cat(upsampled_inputs, dim=1)
        
    def forward(self, x):
        """Forward pass matching official BNHead"""
        # Apply resize_concat transform
        x = self._resize_concat_transform(x)
        
        # Apply batch normalization
        x = self.bn(x)
        
        # Apply segmentation conv
        x = self.conv_seg(x)
        
        return x



class DINOv2Segmenter(nn.Module):
    """DINOv2 Segmenter supporting both linear and multiscale heads"""
    def __init__(self, backbone, segmentation_head, patch_size=14, backbone_type="vitb14", img_ratios=[1.0], head_type="linear"):
        super().__init__()
        self.backbone = backbone
        self.head = segmentation_head
        self.patch_size = patch_size
        self.center_padding = CenterPadding(patch_size)
        self.img_ratios = img_ratios
        self.head_type = head_type
        
        # Set appropriate layer indices for different model sizes
        self.layer_configs = {
            "vits14": [8, 9, 10, 11],    # 12 blocks total
            "vitb14": [8, 9, 10, 11],    # 12 blocks total  
            "vitl14": [20, 21, 22, 23],  # 24 blocks total
            "vitg14": [36, 37, 38, 39],  # 40 blocks total
        }
        if head_type == 'linear':
            # Linear: only use last layer
            self.layer_indices = [self.layer_configs.get(backbone_type, [8, 9, 10, 11])[-1]]
        else:
            # Multiscale: use all 4 layers
            self.layer_indices = self.layer_configs.get(backbone_type, [8, 9, 10, 11])
        
    def forward(self, x):
        """
        Multiscale inference similar to the DINOv2 official implementation
        """
        original_size = x.shape[-2:]
        
        # Store predictions from all scales
        all_predictions = []
        
        for scale in self.img_ratios:
            # Resize input image to current scale
            if scale != 1.0:
                new_h = int(original_size[0] * scale)
                new_w = int(original_size[1] * scale)
                scaled_x = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
            else:
                scaled_x = x
            
            # Apply center padding
            padded_x = self.center_padding(scaled_x)
            
            # Get intermediate features from backbone
            features = self.backbone.get_intermediate_layers(padded_x, n=self.layer_indices, reshape=True)
            
            # Apply segmentation head
            logits = self.head(features)
            
            # Resize logits back to original size
            if logits.shape[-2:] != original_size:
                logits = F.interpolate(logits, size=original_size, mode='bilinear', align_corners=False)
            
            all_predictions.append(logits)
        
        # Ensemble predictions from all scales by averaging
        final_logits = torch.stack(all_predictions, dim=0).mean(dim=0)
        
        return final_logits


class DINOv2SegmentationNode(Node):
    """
    DINOv2 ROS2 Node for Multiscale Semantic Segmentation
    Supports both static image demo and RealSense camera input based on demo_type parameter
    """
    def __init__(self):
        super().__init__('dinov2_segmentation_node')
        
        # ROS2 setup
        self.bridge = CvBridge()
        
        # QoS profile for image publishing (no caching)
        image_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1  # Keep only the latest message
        )
        
        # Parameters (RealSense only mode)
        self.declare_parameter('backbone_size', 'base')  # base or giant
        self.declare_parameter('dataset', 'ade20k')  # 'ade20k' or 'voc2012'
        self.declare_parameter('head_type', 'linear')  # 'linear' or 'multiscale'
        self.declare_parameter('resolution', 518)  # Multiple of 14 for DINOv2 patches
        self.declare_parameter('filter_mode', 'all_classes')  # 'all_classes' or 'lab_only'
        self.declare_parameter('output_mode', 'both')  # '2d', '3d', or 'both'
        
        # Get parameters
        backbone_size = self.get_parameter('backbone_size').get_parameter_value().string_value
        dataset = self.get_parameter('dataset').get_parameter_value().string_value
        head_type = self.get_parameter('head_type').get_parameter_value().string_value
        resolution = self.get_parameter('resolution').get_parameter_value().integer_value
        filter_mode = self.get_parameter('filter_mode').get_parameter_value().string_value
        output_mode = self.get_parameter('output_mode').get_parameter_value().string_value
        
        # Set img_ratios based on head_type
        if head_type == 'linear':
            img_ratios = [1.0]  # Linear probing: single scale for speed
        else:  # multiscale
            img_ratios = FAST  # Multiscale: [1.0, 1.32]
        
        # Store configuration
        self.head_type = head_type
        self.resolution = resolution
        self.filter_mode = filter_mode
        self.output_mode = output_mode
        
        # Setup publishers based on output mode
        if output_mode in ['2d', 'both']:
            self.result_publisher = self.create_publisher(
                Image, '/dinov2/realsense_segmentation_result', image_qos)
        
        if output_mode in ['3d', 'both']:
            self.pointcloud_publisher = self.create_publisher(
                PointCloud2, '/dinov2/realsense_pointcloud', image_qos)
        
        # Setup synchronized subscribers for RGB-D
        if output_mode in ['3d', 'both']:
            # Use message filters for synchronization
            self.rgb_sub = Subscriber(self, Image, '/camera/camera/color/image_raw')
            self.depth_sub = Subscriber(self, Image, '/camera/camera/aligned_depth_to_color/image_raw')
            self.camera_info_sub = Subscriber(self, CameraInfo, '/camera/camera/color/camera_info')
            
            # Synchronizer for RGB-D
            self.sync = ApproximateTimeSynchronizer(
                [self.rgb_sub, self.depth_sub, self.camera_info_sub],
                queue_size=10,
                slop=0.1  # 100ms tolerance
            )
            self.sync.registerCallback(self.rgbd_callback)
            
        else:
            # 2D only mode - simple RGB subscriber
            self.image_subscription = self.create_subscription(
                Image, '/camera/camera/color/image_raw', 
                self.realsense_callback, image_qos)
        
        # Dataset colormap for visualization
        self.colormap = np.array(ADE20K_COLORMAP, dtype=np.uint8)
        
        # Message sequence counter for better synchronization
        self.message_seq = 0
        
        # Initialize model
        self.get_logger().info(f"Loading DINOv2 {backbone_size} model with {head_type} head...")
        self.get_logger().info(f"Using scales: {img_ratios}")
        self.get_logger().info(f"Using resolution: {resolution}")
        self.get_logger().info(f"Using dataset: {dataset}")
        self.get_logger().info(f"Filter mode: {filter_mode}")
        self.load_model(backbone_size, img_ratios, dataset, head_type)
        
        # Initialize camera intrinsics (will be updated from camera_info)
        self.camera_matrix = None
        self.dist_coeffs = None
        
        self.get_logger().info("3D visualization available via RViz - topic: /dinov2/realsense_pointcloud")
        
        self.get_logger().info("Waiting for RealSense camera images on /camera/camera/color/image_raw")
        self.get_logger().info(f"DINOv2 Segmentation Node initialized (mode: {output_mode})!")
    
    def letterbox_resize(self, pil_image, target_size):
        """
        Letterbox resize that preserves aspect ratio and adds padding
        Similar to DINOv2 official preprocessing
        """
        original_width, original_height = pil_image.size
        
        # Calculate scaling factor to fit within target_size
        scale = min(target_size / original_width, target_size / original_height)
        
        # Calculate new dimensions
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        # Resize image maintaining aspect ratio
        resized_image = pil_image.resize((new_width, new_height), PILImage.Resampling.BILINEAR)
        
        # Create new image with target size and fill with gray padding
        letterbox_image = PILImage.new('RGB', (target_size, target_size), (128, 128, 128))
        
        # Calculate padding offsets to center the image
        paste_x = (target_size - new_width) // 2
        paste_y = (target_size - new_height) // 2
        
        # Paste resized image onto padded background
        letterbox_image.paste(resized_image, (paste_x, paste_y))
        
        # Store padding info for later use in post-processing
        self.padding_info = {
            'scale': scale,
            'paste_x': paste_x,
            'paste_y': paste_y,
            'new_width': new_width,
            'new_height': new_height,
            'original_width': original_width,
            'original_height': original_height
        }
        
        return letterbox_image
    
    def postprocess_segmentation(self, segmentation_logits, original_size):
        """
        Remove padding and resize segmentation back to original dimensions
        """
        # Extract segmentation from padded region
        pad_info = self.padding_info
        paste_x, paste_y = pad_info['paste_x'], pad_info['paste_y']
        new_width, new_height = pad_info['new_width'], pad_info['new_height']
        
        # Crop out the actual image region (remove padding)
        segmentation_cropped = segmentation_logits[
            paste_y:paste_y + new_height,
            paste_x:paste_x + new_width
        ]
        
        # Resize back to original dimensions
        segmentation_resized = cv2.resize(
            segmentation_cropped.astype(np.uint8),
            original_size,  # (width, height) format for cv2
            interpolation=cv2.INTER_NEAREST
        )
        
        return segmentation_resized.astype(np.int32)
        
    def download_pretrained_head(self, backbone_name, dataset="ade20k", head_type="ms"):
        """Download pretrained segmentation head from DINOv2 official repository"""
        try:
            models_dir = os.path.join(os.path.dirname(__file__), '..', 'models', 'pretrained')
            os.makedirs(models_dir, exist_ok=True)
            
            # DINOv2 official model URLs
            base_url = "https://dl.fbaipublicfiles.com/dinov2"
            checkpoint_url = f"{base_url}/{backbone_name}/{backbone_name}_{dataset}_{head_type}_head.pth"
            
            checkpoint_file = os.path.join(models_dir, f"{backbone_name}_{dataset}_{head_type}_head.pth")
            
            # Download if not exists
            if not os.path.exists(checkpoint_file):
                self.get_logger().info(f"📥 Downloading pretrained weights from: {checkpoint_url}")
                urllib.request.urlretrieve(checkpoint_url, checkpoint_file)
                self.get_logger().info(f"✅ Downloaded to: {checkpoint_file}")
            else:
                self.get_logger().info(f"📁 Using cached pretrained weights: {checkpoint_file}")
            
            return checkpoint_file
            
        except Exception as e:
            self.get_logger().error(f"❌ Failed to download pretrained weights: {str(e)}")
            raise e
    
    def load_model(self, backbone_size='base', img_ratios=FAST, dataset='ade20k', head_type='linear'):
        """Load DINOv2 model with pretrained multiscale segmentation head without MMCV"""
        try:
            # Backbone configuration
            backbone_archs = {
                "small": "vits14",
                "base": "vitb14", 
                "large": "vitl14",
                "giant": "vitg14",
            }
            backbone_arch = backbone_archs[backbone_size]
            backbone_name = f"dinov2_{backbone_arch}"
            
            # Feature dimensions for different backbones
            feature_dims = {
                "vits14": 384,
                "vitb14": 768, 
                "vitl14": 1024,
                "vitg14": 1536,
            }
            feature_dim = feature_dims[backbone_arch]
            
            # Load backbone with improved stability
            self.get_logger().info(f"Loading backbone: {backbone_name}")
            try:
                backbone_model = torch.hub.load(
                    "facebookresearch/dinov2", 
                    backbone_name,
                    force_reload=False,  # Use cache if available
                    trust_repo=True
                )
                backbone_model.eval()
                self.get_logger().info(f"Successfully loaded {backbone_name} via torch.hub")
            except Exception as e:
                self.get_logger().error(f"Failed to load backbone via torch.hub: {e}")
                raise e
            
            # Setup device with FP16 optimization
            if torch.cuda.is_available():
                backbone_model.cuda()
                # Enable FP16 for performance
                backbone_model.half()
                self.device = 'cuda'
                self.use_fp16 = True
                self.get_logger().info("Using CUDA with FP16 optimization")
            else:
                self.device = 'cpu'
                self.use_fp16 = False
                self.get_logger().info("Using CPU (FP16 disabled)")
            
            # Create segmentation head based on head_type
            num_classes = 150 if dataset == 'ade20k' else 21  # VOC2012 has 21 classes
            
            if head_type == 'linear':
                # Linear head: single layer from last backbone layer only
                seg_head = LinearSegmentationHead(
                    in_channels=feature_dim, 
                    num_classes=num_classes
                )
            else:  # multiscale
                # Use official BNHead for compatibility with pretrained weights
                seg_head = BNHead(
                    num_classes=num_classes,
                    backbone_size=backbone_size  # Pass backbone size for correct channels
                )
            
            # Create complete segmenter with correct backbone type
            model = DINOv2Segmenter(
                backbone_model, 
                seg_head, 
                patch_size=14, 
                backbone_type=backbone_arch,
                img_ratios=img_ratios,
                head_type=head_type
            )
            model.to(self.device)
            model.eval()
            
            # Apply FP16 to segmentation head as well
            if self.use_fp16:
                model.head.half()
            
            self.get_logger().info(f"Using layer indices {model.layer_indices} for {backbone_arch}")
            self.get_logger().info(f"Using multiscale ratios: {img_ratios}")
            
            # Download and load pretrained head weights from DINOv2
            try:
                checkpoint_file = self.download_pretrained_head(backbone_name, dataset)
                checkpoint = torch.load(checkpoint_file, map_location=self.device)
                
                # Extract segmentation head weights
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                
                # Load only the segmentation head weights with proper key mapping
                head_state_dict = {}
                self.get_logger().info(f"🔍 Raw state_dict keys: {list(state_dict.keys())[:10]}")
                
                for key, value in state_dict.items():
                    if key.startswith('decode_head.'):
                        new_key = key.replace('decode_head.', '')
                        
                        # Debug: 모든 키 변환 과정 출력
                        self.get_logger().info(f"🔧 Key mapping: {key} -> {new_key}")
                        
                        # 정확한 키 매핑 (MMSeg BNHead 기준)
                        if 'conv_seg.' in new_key or 'cls_seg.' in key:
                            # conv_seg 또는 cls_seg -> conv_seg
                            mapped_key = new_key.replace('cls_seg.', 'conv_seg.')
                            head_state_dict[mapped_key] = value
                        elif 'bn.' in new_key or 'batch_norm.' in new_key:
                            # BatchNorm 키 정규화
                            mapped_key = new_key.replace('batch_norm.', 'bn.')
                            head_state_dict[mapped_key] = value
                        else:
                            # 기타 키들
                            head_state_dict[new_key] = value
                
                # Debug: Print available keys in checkpoint
                self.get_logger().info(f"🔍 Available weight keys: {list(head_state_dict.keys())[:5]}...")
                self.get_logger().info(f"🔍 Our head structure keys: {list(model.head.state_dict().keys())}")
                
                # Load the head weights with strict=False to handle key mismatches gracefully
                missing_keys, unexpected_keys = model.head.load_state_dict(head_state_dict, strict=False)
                
                if missing_keys:
                    self.get_logger().error(f"❌ Missing keys in head weights: {missing_keys}")
                if unexpected_keys:
                    self.get_logger().error(f"❌ Unexpected keys in head weights: {unexpected_keys}")
                    
                # Check if weights were actually loaded
                loaded_params = sum(1 for key in head_state_dict.keys() if key in model.head.state_dict())
                total_params = len(model.head.state_dict())
                
                if loaded_params == 0:
                    self.get_logger().error("❌ NO PRETRAINED WEIGHTS LOADED! Using random weights!")
                elif loaded_params < total_params:
                    self.get_logger().warn(f"⚠️ Partial weight loading: {loaded_params}/{total_params}")
                else:
                    self.get_logger().info("✅ Successfully loaded pretrained DINOv2 BNHead weights!")
                    
                self.get_logger().info(f"Loaded {loaded_params}/{total_params} weight parameters")
                
            except Exception as e:
                self.get_logger().warn(f"⚠️ Could not load pretrained weights: {str(e)}")
                self.get_logger().info("Using randomly initialized segmentation head")
            
            self.model = model
            self.get_logger().info(f"Successfully loaded {backbone_size} multiscale model!")
            
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {str(e)}")
            import traceback
            traceback.print_exc()
            raise e
    
    def filter_lab_objects(self, segmentation_logits):
        """Filter segmentation to show only lab-related objects"""
        filtered_segmentation = np.zeros_like(segmentation_logits)
        
        # Keep only lab object classes
        for class_id in LAB_OBJECT_CLASSES.keys():
            mask = segmentation_logits == class_id
            filtered_segmentation[mask] = class_id
            
        return filtered_segmentation
    
    def render_segmentation(self, segmentation_logits, original_image=None, filter_lab_objects=True):
        """Render segmentation result with overlay on original image"""
        # Filter to show only lab objects if requested
        if filter_lab_objects:
            filtered_logits = self.filter_lab_objects(segmentation_logits)
        else:
            filtered_logits = segmentation_logits
        
        if original_image is not None:
            # Create overlay of segmentation on original image
            original_array = np.array(original_image)
            overlay_image = original_array.copy()
            
            # Handle array bounds
            seg_indices = np.clip(filtered_logits + 1, 0, len(self.colormap) - 1)
            
            # Create mask for detected objects (non-background)
            object_mask = filtered_logits > 0
            
            if np.any(object_mask):
                # Apply segmentation colors only where objects are detected
                seg_colors = self.colormap[seg_indices]
                
                # Blend: 30% original + 70% segmentation color for detected objects (more visible)
                overlay_image[object_mask] = (
                    original_array[object_mask] * 0.3 + 
                    seg_colors[object_mask] * 0.7
                ).astype(np.uint8)
            
            return overlay_image
        else:
            # No original image, just return segmentation colors
            seg_indices = np.clip(filtered_logits + 1, 0, len(self.colormap) - 1)
            return self.colormap[seg_indices].astype(np.uint8)
    
    def process_image(self, pil_image, source_type="unknown"):
        """Process image with DINOv2 multiscale segmentation without MMCV"""
        if pil_image is None:
            return
        
        try:
            original_size = pil_image.size
            self.get_logger().info(f"Processing {source_type} image with multiscale inference: {original_size}")
            
            # Letterbox resize (preserve aspect ratio with padding)
            pil_image_processed = self.letterbox_resize(pil_image, self.resolution)
            self.get_logger().info(f"Letterboxed to: {pil_image_processed.size}")
            
            # Prepare image tensor
            image_array = np.array(pil_image_processed).astype(np.float32) / 255.0
            
            # Convert to tensor and normalize (ImageNet normalization)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image_array = (image_array - mean) / std
            
            # Convert to tensor: HWC -> CHW and add batch dimension
            image_tensor = torch.from_numpy(image_array.transpose(2, 0, 1)).unsqueeze(0)
            if self.use_fp16:
                image_tensor = image_tensor.half().to(self.device)
            else:
                image_tensor = image_tensor.float().to(self.device)
            
            # Perform multiscale inference
            with torch.no_grad():
                self.get_logger().info(f"Running inference with scales: {self.model.img_ratios}")
                logits = self.model(image_tensor)
                
                # Get predictions
                segmentation_logits = logits.argmax(dim=1).squeeze(0).cpu().numpy()
            
            # Remove padding and resize back to original size
            segmentation_logits_resized = self.postprocess_segmentation(segmentation_logits, original_size)
            
            # 필터링 모드에 따라 클래스 필터링 결정
            if self.filter_mode == 'lab_only':
                filter_objects = True  # LAB 클래스만
            else:  # 'all_classes'
                filter_objects = False  # 모든 클래스
            
            segmented_image = self.render_segmentation(segmentation_logits_resized, pil_image, filter_lab_objects=filter_objects)
            
            # Convert PIL to OpenCV format for publishing
            image_array = np.array(pil_image)
            original_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            
            # Add slight noise to ensure image is different (invisible to human eye)
            noise = np.random.randint(-1, 2, original_bgr.shape, dtype=np.int8)
            original_bgr = np.clip(original_bgr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            result_bgr = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
            noise_seg = np.random.randint(-1, 2, result_bgr.shape, dtype=np.int8) 
            result_bgr = np.clip(result_bgr.astype(np.int16) + noise_seg, 0, 255).astype(np.uint8)
            
            # Use same timestamp for perfect synchronization
            timestamp = self.get_clock().now().to_msg()
            self.message_seq += 1
            
            # Create and publish segmentation result
            try:
                result_msg = self.bridge.cv2_to_imgmsg(result_bgr, "bgr8")
                result_msg.header.stamp = timestamp
                
                # Set frame_id for RealSense
                frame_id = "camera_color_optical_frame"
                
                # Force unique message ID to prevent caching
                import time
                unique_id = str(int(time.time() * 1000000))  # microsecond timestamp
                result_msg.header.frame_id = f"{frame_id}_{unique_id}"
                
                # Publish segmentation result
                self.result_publisher.publish(result_msg)
                
                self.get_logger().info(f"📸 Published {source_type} multiscale images #{self.message_seq} (ID: {unique_id[-6:]})")
                
            except Exception as e:
                self.get_logger().error(f"Failed to publish images: {str(e)}")
            
            # Print statistics - show ALL detected classes first for debugging
            all_unique_classes = np.unique(segmentation_logits_resized)
            all_detected = []
            for class_id in all_unique_classes:
                if class_id > 0:  # Skip background
                    pixel_count = np.sum(segmentation_logits_resized == class_id)
                    percentage = (pixel_count / segmentation_logits_resized.size) * 100
                    if percentage > 0.1:  # Only show classes with >0.1% pixels
                        class_name = ADE20K_CLASSES[class_id] if class_id < len(ADE20K_CLASSES) else f"Unknown{class_id}"
                        all_detected.append(f"Class {class_id}({class_name}): {pixel_count}px ({percentage:.1f}%)")
            
            self.get_logger().info(f"🔍 ALL detected classes: {', '.join(all_detected[:5])}")  # Show top 5
            
            # Print lab object statistics with detailed debugging
            self.get_logger().info(f"🔧 DEBUG - LAB_OBJECT_CLASSES keys: {list(LAB_OBJECT_CLASSES.keys())}")
            
            filtered_logits = self.filter_lab_objects(segmentation_logits_resized)
            
            # Debug: 필터링 전후 비교
            original_unique = np.unique(segmentation_logits_resized)
            filtered_unique = np.unique(filtered_logits)
            self.get_logger().info(f"🔧 DEBUG - Original unique classes: {original_unique[original_unique > 0]}")
            self.get_logger().info(f"🔧 DEBUG - Filtered unique classes: {filtered_unique[filtered_unique > 0]}")
            
            unique_classes = np.unique(filtered_logits)
            detected_objects = []
            
            for class_id in unique_classes:
                if class_id > 0:  # 모든 non-background 클래스 확인
                    pixel_count = np.sum(filtered_logits == class_id)
                    percentage = (pixel_count / filtered_logits.size) * 100
                    
                    # LAB_OBJECT_CLASSES에 있는지 확인
                    if class_id in LAB_OBJECT_CLASSES:
                        color_idx = int(class_id + 1)
                        if color_idx < len(self.colormap):
                            r, g, b = self.colormap[color_idx]
                            detected_objects.append(f"{class_id}({LAB_OBJECT_CLASSES[class_id]}): {pixel_count}px ({percentage:.1f}%) RGB({r},{g},{b})")
                    else:
                        self.get_logger().info(f"🔧 DEBUG - Class {class_id} NOT in LAB_OBJECT_CLASSES but found in filtered: {pixel_count}px")
            
            self.get_logger().info(f"🎯 Detected {len(detected_objects)} lab objects:")
            for obj in detected_objects:
                self.get_logger().info(f"  - {obj}")
            
            self.get_logger().info(f"✅ {source_type} Multiscale Segmentation completed and published!")
            
        except Exception as e:
            self.get_logger().error(f"Image processing failed: {str(e)}")
            import traceback
            traceback.print_exc()
    
    
    def rgbd_callback(self, rgb_msg, depth_msg, camera_info_msg):
        """Handle synchronized RGB-D callback for 3D processing"""
        try:
            # Update camera intrinsics
            self.update_camera_intrinsics(camera_info_msg)
            
            # Convert ROS messages to OpenCV
            rgb_cv = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            depth_cv = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
            
            # Convert BGR to RGB for processing
            rgb_image = cv2.cvtColor(rgb_cv, cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(rgb_image)
            
            # Run 2D segmentation
            segmentation_logits = self.run_segmentation_inference(pil_image)
            
            # Apply filtering
            if self.filter_mode == 'lab_only':
                segmentation_logits = self.filter_lab_objects(segmentation_logits)
            
            # Publish outputs based on mode
            if self.output_mode in ['2d', 'both']:
                self.publish_2d_result(segmentation_logits, pil_image)
            
            if self.output_mode in ['3d', 'both']:
                self.publish_3d_result(segmentation_logits, rgb_image, depth_cv)
                
        except Exception as e:
            self.get_logger().error(f"Failed to process RGB-D: {str(e)}")
    
    def realsense_callback(self, msg):
        """Handle RealSense RGB Image input (2D only mode)"""
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # Convert to PIL Image
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(rgb_image)
            self.process_image(pil_image, "RealSense")
        except Exception as e:
            self.get_logger().error(f"Failed to process RealSense Image: {str(e)}")
    
    def update_camera_intrinsics(self, camera_info_msg):
        """Update camera intrinsics from CameraInfo message"""
        if self.camera_matrix is None:
            self.camera_matrix = np.array(camera_info_msg.k).reshape(3, 3)
            self.dist_coeffs = np.array(camera_info_msg.d)
            self.get_logger().info(f"Updated camera intrinsics: fx={self.camera_matrix[0,0]:.1f}, fy={self.camera_matrix[1,1]:.1f}")
    
    def run_segmentation_inference(self, pil_image):
        """Run segmentation inference and return class map"""
        original_size = pil_image.size
        
        # Letterbox resize
        pil_image_processed = self.letterbox_resize(pil_image, self.resolution)
        
        # Prepare tensor
        image_array = np.array(pil_image_processed).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_array = (image_array - mean) / std
        
        image_tensor = torch.from_numpy(image_array.transpose(2, 0, 1)).unsqueeze(0)
        if self.use_fp16:
            image_tensor = image_tensor.half().to(self.device)
        else:
            image_tensor = image_tensor.float().to(self.device)
        
        # Inference
        with torch.no_grad():
            logits = self.model(image_tensor)
            segmentation_logits = logits.argmax(dim=1).squeeze(0).cpu().numpy()
        
        # Post-process back to original size
        segmentation_logits_resized = self.postprocess_segmentation(segmentation_logits, original_size)
        return segmentation_logits_resized
    
    def publish_2d_result(self, segmentation_logits, pil_image):
        """Publish 2D segmentation result"""
        segmented_image = self.render_segmentation(segmentation_logits, pil_image, filter_lab_objects=False)
        
        # Convert to BGR for ROS
        result_bgr = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
        
        # Create and publish message
        result_msg = self.bridge.cv2_to_imgmsg(result_bgr, "bgr8")
        result_msg.header.stamp = self.get_clock().now().to_msg()
        result_msg.header.frame_id = "camera_color_optical_frame"
        
        if hasattr(self, 'result_publisher'):
            self.result_publisher.publish(result_msg)
    
    def publish_3d_result(self, segmentation_logits, rgb_image, depth_image):
        """Generate and publish 3D point cloud with segmentation colors"""
        if self.camera_matrix is None:
            self.get_logger().warn("Camera intrinsics not available, skipping 3D output")
            return
        
        # Create point cloud
        points_3d, colors = self.create_colored_pointcloud(segmentation_logits, rgb_image, depth_image)
        
        if len(points_3d) > 0:
            # Create PointCloud2 message
            pointcloud_msg = self.create_pointcloud2_msg(points_3d, colors)
            self.pointcloud_publisher.publish(pointcloud_msg)
            
            # 3D data published to RViz (Open3D visualization removed)
            
            self.get_logger().info(f"Published 3D point cloud with {len(points_3d)} points")
    
    def create_colored_pointcloud(self, segmentation_logits, rgb_image, depth_image):
        """Convert RGB-D + segmentation to colored 3D points"""
        # Get dimensions
        seg_height, seg_width = segmentation_logits.shape
        depth_height, depth_width = depth_image.shape
        rgb_height, rgb_width = rgb_image.shape[:2]
        
        self.get_logger().info(f"Dimensions - RGB: {rgb_width}x{rgb_height}, Depth: {depth_width}x{depth_height}, Seg: {seg_width}x{seg_height}")
        
        # Resize depth and segmentation to match RGB dimensions
        if (depth_height, depth_width) != (rgb_height, rgb_width):
            depth_image_resized = cv2.resize(depth_image, (rgb_width, rgb_height), interpolation=cv2.INTER_NEAREST)
        else:
            depth_image_resized = depth_image
            
        if (seg_height, seg_width) != (rgb_height, rgb_width):
            segmentation_resized = cv2.resize(segmentation_logits.astype(np.uint8), (rgb_width, rgb_height), interpolation=cv2.INTER_NEAREST)
        else:
            segmentation_resized = segmentation_logits
        
        # Use RGB dimensions as reference
        height, width = rgb_height, rgb_width
        fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
        cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]
        
        # Create coordinate matrices
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        
        # Filter valid depth points (depth > 0 and < 5000mm)
        valid_mask = (depth_image_resized > 0) & (depth_image_resized < 5000)
        
        u_valid = u[valid_mask]
        v_valid = v[valid_mask]
        depth_valid = depth_image_resized[valid_mask].astype(np.float32) / 1000.0  # Convert to meters
        
        # Convert to 3D coordinates
        x = (u_valid - cx) * depth_valid / fx
        y = (v_valid - cy) * depth_valid / fy
        z = depth_valid
        
        points_3d = np.column_stack((x, y, z))
        
        # Get blended colors (RGB + segmentation overlay)
        rgb_colors = rgb_image[valid_mask]  # Original RGB colors
        seg_colors = self.get_segmentation_colors(segmentation_resized[valid_mask])  # Segmentation colors
        blended_colors = self.blend_rgb_segmentation(rgb_colors, seg_colors, segmentation_resized[valid_mask])
        
        return points_3d, blended_colors
    
    def get_segmentation_colors(self, class_indices):
        """Convert class indices to RGB colors"""
        colors = np.zeros((len(class_indices), 3), dtype=np.uint8)
        
        for i, class_id in enumerate(class_indices):
            if class_id >= 0 and class_id < len(self.colormap):
                colors[i] = self.colormap[class_id + 1]  # +1 for colormap indexing
            else:
                colors[i] = [128, 128, 128]  # Gray for unknown classes
        
        return colors
    
    def blend_rgb_segmentation(self, rgb_colors, seg_colors, class_indices):
        """Blend RGB colors with segmentation colors (like rqt_image_view overlay)"""
        blended_colors = np.zeros_like(rgb_colors, dtype=np.uint8)
        
        # Background (class_id == 0): use original RGB
        background_mask = class_indices == 0
        blended_colors[background_mask] = rgb_colors[background_mask]
        
        # Segmented objects: blend RGB + segmentation (30% RGB + 70% segmentation)
        object_mask = class_indices > 0
        if np.any(object_mask):
            blended_colors[object_mask] = (
                rgb_colors[object_mask] * 0.3 + 
                seg_colors[object_mask] * 0.7
            ).astype(np.uint8)
        
        return blended_colors
    
    def create_pointcloud2_msg(self, points, colors):
        """Create PointCloud2 message from points and colors with packed RGB"""
        # Define fields for XYZ + packed RGB format (RViz2 compatible)
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
        ]
        
        # Pack RGB values into uint32 (0xRRGGBB format)
        rgb_packed = (colors[:, 0].astype(np.uint32) << 16) | \
                     (colors[:, 1].astype(np.uint32) << 8) | \
                     (colors[:, 2].astype(np.uint32))
        
        # Prepare structured data
        num_points = len(points)
        point_cloud_data = np.zeros(num_points, dtype=[
            ('x', np.float32),
            ('y', np.float32), 
            ('z', np.float32),
            ('rgb', np.uint32)
        ])
        
        # Fill data
        point_cloud_data['x'] = points[:, 0]
        point_cloud_data['y'] = points[:, 1] 
        point_cloud_data['z'] = points[:, 2]
        point_cloud_data['rgb'] = rgb_packed
        
        # Create message
        msg = PointCloud2()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera_color_optical_frame"
        msg.height = 1
        msg.width = num_points
        msg.fields = fields
        msg.is_bigendian = False
        msg.point_step = 16  # 3*4 + 1*4 = 16 bytes per point
        msg.row_step = msg.point_step * msg.width
        msg.data = point_cloud_data.tobytes()
        msg.is_dense = False  # May contain NaN/Inf values
        
        return msg
    
    # Open3D functions removed - using RViz only for 3D visualization
    


def main(args=None):
    rclpy.init(args=args)
    
    node = DINOv2SegmentationNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()