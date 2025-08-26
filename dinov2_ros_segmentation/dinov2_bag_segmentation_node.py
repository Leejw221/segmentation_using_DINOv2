#!/usr/bin/env python3

import os
import math
import itertools
import urllib.request

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2
import numpy as np
from PIL import Image as PILImage

import torch
import torch.nn as nn
import torch.nn.functional as F

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

# ADE20K Official Colormap
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

# Extend colormap to 151 classes
while len(ADE20K_COLORMAP) < 151:
    ADE20K_COLORMAP.append((128, 128, 128))

# Lab objects class IDs for filtering
LAB_OBJECT_CLASSES = {
    20: "chair",           # 의자
    28: "mirror",          # 거울
    42: "box",             # 상자
    68: "book",            # 책
    99: "bottle",          # 병
    120: "ball",           # 공
    144: "monitor",        # 모니터
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


class BNHead(nn.Module):
    def __init__(self, num_classes=150, backbone_size='base'):
        super().__init__()
        hardcoded_channels = {
            'small': 384 * 4,
            'base': 768 * 4,
            'large': 1024 * 4,
            'giant': 1536 * 4
        }
        
        self.in_channels = hardcoded_channels.get(backbone_size, 768 * 4)
        
        try:
            self.bn = nn.SyncBatchNorm(self.in_channels)
        except:
            self.bn = nn.BatchNorm2d(self.in_channels)
            
        self.conv_seg = nn.Conv2d(self.in_channels, num_classes, kernel_size=1)
        self.align_corners = False
        
    def _resize_concat_transform(self, inputs):
        if not isinstance(inputs, (list, tuple)):
            return inputs
        
        input_list = []
        for x in inputs:
            if isinstance(x, list):
                input_list.extend(x)
            else:
                input_list.append(x)
        inputs = input_list
        
        for i, x in enumerate(inputs):
            if len(x.shape) == 2:
                inputs[i] = x[:, :, None, None]
        
        if len(inputs) == 0:
            return None
            
        target_h, target_w = inputs[0].shape[-2:]
        
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
        
        return torch.cat(upsampled_inputs, dim=1)
        
    def forward(self, x):
        x = self._resize_concat_transform(x)
        x = self.bn(x)
        x = self.conv_seg(x)
        return x


class DINOv2Segmenter(nn.Module):
    def __init__(self, backbone, segmentation_head, patch_size=14, backbone_type="vitb14", img_ratios=[1.0]):
        super().__init__()
        self.backbone = backbone
        self.head = segmentation_head
        self.patch_size = patch_size
        self.center_padding = CenterPadding(patch_size)
        self.img_ratios = img_ratios
        
        layer_configs = {
            "vits14": [8, 9, 10, 11],
            "vitb14": [8, 9, 10, 11],  
            "vitl14": [20, 21, 22, 23],
            "vitg14": [36, 37, 38, 39],
        }
        self.layer_indices = layer_configs.get(backbone_type, [8, 9, 10, 11])
        
    def forward(self, x):
        original_size = x.shape[-2:]
        all_predictions = []
        
        for scale in self.img_ratios:
            if scale != 1.0:
                new_h = int(original_size[0] * scale)
                new_w = int(original_size[1] * scale)
                scaled_x = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
            else:
                scaled_x = x
            
            padded_x = self.center_padding(scaled_x)
            features = self.backbone.get_intermediate_layers(padded_x, n=self.layer_indices, reshape=True)
            logits = self.head(features)
            
            if logits.shape[-2:] != original_size:
                logits = F.interpolate(logits, size=original_size, mode='bilinear', align_corners=False)
            
            all_predictions.append(logits)
        
        final_logits = torch.stack(all_predictions, dim=0).mean(dim=0)
        return final_logits


class DINOv2BagSegmentationNode(Node):
    """
    Dedicated bag segmentation node optimized for GPU processing
    Subscribes to bag-played image topics and publishes segmentation results
    """
    def __init__(self):
        super().__init__('dinov2_bag_segmentation_node')
        
        self.bridge = CvBridge()
        
        # QoS profile optimized for bag playback
        image_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5  # Larger buffer for bag playback
        )
        
        # Parameters
        self.declare_parameter('backbone_size', 'base')
        self.declare_parameter('head_type', 'multiscale')
        self.declare_parameter('resolution', 518)
        self.declare_parameter('filter_mode', 'all_classes')
        self.declare_parameter('input_topic', '/camera/color/image_raw')  # Original bag topic
        
        backbone_size = self.get_parameter('backbone_size').get_parameter_value().string_value
        head_type = self.get_parameter('head_type').get_parameter_value().string_value
        resolution = self.get_parameter('resolution').get_parameter_value().integer_value
        filter_mode = self.get_parameter('filter_mode').get_parameter_value().string_value
        input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        
        self.resolution = resolution
        self.filter_mode = filter_mode
        
        # Set up subscribers and publishers
        self.image_subscription = self.create_subscription(
            Image, input_topic, self.image_callback, image_qos)
            
        self.result_publisher = self.create_publisher(
            Image, '/dinov2/bag_segmentation_result', image_qos)
        
        # Dataset colormap
        self.colormap = np.array(ADE20K_COLORMAP, dtype=np.uint8)
        
        # Performance counters
        self.message_count = 0
        self.processing_times = []
        
        # Load model with GPU optimization
        self.get_logger().info(f"🚀 Loading DINOv2 {backbone_size} model with {head_type} head for bag processing...")
        self.get_logger().info(f"📺 GPU Info: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU'}")
        self.load_model(backbone_size, head_type)
        
        self.get_logger().info(f"🎯 Waiting for bag images on {input_topic}")
        self.get_logger().info("🎥 DINOv2 Bag Segmentation Node ready!")
    
    def letterbox_resize(self, pil_image, target_size):
        original_width, original_height = pil_image.size
        scale = min(target_size / original_width, target_size / original_height)
        
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        resized_image = pil_image.resize((new_width, new_height), PILImage.Resampling.BILINEAR)
        letterbox_image = PILImage.new('RGB', (target_size, target_size), (128, 128, 128))
        
        paste_x = (target_size - new_width) // 2
        paste_y = (target_size - new_height) // 2
        letterbox_image.paste(resized_image, (paste_x, paste_y))
        
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
        pad_info = self.padding_info
        paste_x, paste_y = pad_info['paste_x'], pad_info['paste_y']
        new_width, new_height = pad_info['new_width'], pad_info['new_height']
        
        segmentation_cropped = segmentation_logits[
            paste_y:paste_y + new_height,
            paste_x:paste_x + new_width
        ]
        
        segmentation_resized = cv2.resize(
            segmentation_cropped.astype(np.uint8),
            original_size,
            interpolation=cv2.INTER_NEAREST
        )
        
        return segmentation_resized.astype(np.int32)
    
    def download_pretrained_head(self, backbone_name):
        try:
            models_dir = os.path.join(os.path.dirname(__file__), '..', 'models', 'pretrained')
            os.makedirs(models_dir, exist_ok=True)
            
            base_url = "https://dl.fbaipublicfiles.com/dinov2"
            checkpoint_url = f"{base_url}/{backbone_name}/{backbone_name}_ade20k_ms_head.pth"
            checkpoint_file = os.path.join(models_dir, f"{backbone_name}_ade20k_ms_head.pth")
            
            if not os.path.exists(checkpoint_file):
                self.get_logger().info(f"📥 Downloading pretrained weights...")
                urllib.request.urlretrieve(checkpoint_url, checkpoint_file)
                self.get_logger().info("✅ Download complete!")
            else:
                self.get_logger().info("📁 Using cached weights")
            
            return checkpoint_file
        except Exception as e:
            self.get_logger().error(f"❌ Failed to download weights: {str(e)}")
            raise e
    
    def load_model(self, backbone_size='base', head_type='multiscale'):
        try:
            backbone_archs = {
                "small": "vits14",
                "base": "vitb14", 
                "large": "vitl14",
                "giant": "vitg14",
            }
            backbone_arch = backbone_archs[backbone_size]
            backbone_name = f"dinov2_{backbone_arch}"
            
            # Load backbone
            self.get_logger().info(f"🔧 Loading backbone: {backbone_name}")
            backbone_model = torch.hub.load(
                "facebookresearch/dinov2", 
                backbone_name,
                force_reload=False,
                trust_repo=True
            )
            backbone_model.eval()
            
            # Setup GPU with proper optimization
            if torch.cuda.is_available():
                backbone_model.cuda()
                backbone_model.half()  # FP16 for speed
                self.device = 'cuda'
                self.use_fp16 = True
                self.get_logger().info(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
                self.get_logger().info(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
            else:
                self.device = 'cpu'
                self.use_fp16 = False
                self.get_logger().warn("⚠️ GPU not available, using CPU")
            
            # Create segmentation head
            img_ratios = [1.0, 1.25] if head_type == 'multiscale' else [1.0]
            seg_head = BNHead(num_classes=150, backbone_size=backbone_size)
            
            # Create segmenter
            model = DINOv2Segmenter(
                backbone_model, 
                seg_head, 
                patch_size=14, 
                backbone_type=backbone_arch,
                img_ratios=img_ratios
            )
            model.to(self.device)
            model.eval()
            
            if self.use_fp16:
                model.head.half()
            
            # Load pretrained weights
            try:
                checkpoint_file = self.download_pretrained_head(backbone_name)
                checkpoint = torch.load(checkpoint_file, map_location=self.device)
                
                state_dict = checkpoint.get('state_dict', checkpoint)
                head_state_dict = {}
                
                for key, value in state_dict.items():
                    if key.startswith('decode_head.'):
                        new_key = key.replace('decode_head.', '')
                        if 'conv_seg.' in new_key or 'cls_seg.' in key:
                            mapped_key = new_key.replace('cls_seg.', 'conv_seg.')
                            head_state_dict[mapped_key] = value
                        elif 'bn.' in new_key or 'batch_norm.' in new_key:
                            mapped_key = new_key.replace('batch_norm.', 'bn.')
                            head_state_dict[mapped_key] = value
                        else:
                            head_state_dict[new_key] = value
                
                missing_keys, unexpected_keys = model.head.load_state_dict(head_state_dict, strict=False)
                
                if len(head_state_dict) > 0 and not missing_keys:
                    self.get_logger().info("✅ Successfully loaded pretrained weights!")
                else:
                    self.get_logger().warn("⚠️ Using random weights")
                    
            except Exception as e:
                self.get_logger().warn(f"⚠️ Weight loading failed: {str(e)}")
                self.get_logger().info("Using random weights")
            
            self.model = model
            self.get_logger().info(f"🎯 Model loaded successfully with {head_type} head!")
            
        except Exception as e:
            self.get_logger().error(f"❌ Model loading failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise e
    
    def filter_lab_objects(self, segmentation_logits):
        filtered_segmentation = np.zeros_like(segmentation_logits)
        for class_id in LAB_OBJECT_CLASSES.keys():
            mask = segmentation_logits == class_id
            filtered_segmentation[mask] = class_id
        return filtered_segmentation
    
    def render_segmentation(self, segmentation_logits, original_image=None):
        if self.filter_mode == 'lab_only':
            segmentation_logits = self.filter_lab_objects(segmentation_logits)
        
        if original_image is not None:
            original_array = np.array(original_image)
            overlay_image = original_array.copy()
            
            seg_indices = np.clip(segmentation_logits + 1, 0, len(self.colormap) - 1)
            object_mask = segmentation_logits > 0
            
            if np.any(object_mask):
                seg_colors = self.colormap[seg_indices]
                overlay_image[object_mask] = (
                    original_array[object_mask] * 0.3 + 
                    seg_colors[object_mask] * 0.7
                ).astype(np.uint8)
            
            return overlay_image
        else:
            seg_indices = np.clip(segmentation_logits + 1, 0, len(self.colormap) - 1)
            return self.colormap[seg_indices].astype(np.uint8)
    
    def image_callback(self, msg):
        try:
            import time
            start_time = time.time()
            
            # Convert ROS Image to PIL
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(rgb_image)
            
            original_size = pil_image.size
            self.message_count += 1
            
            self.get_logger().info(f"🖼️ Processing bag image #{self.message_count}: {original_size}")
            
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
            
            # GPU inference
            with torch.no_grad():
                logits = self.model(image_tensor)
                segmentation_logits = logits.argmax(dim=1).squeeze(0).cpu().numpy()
            
            # Post-process
            segmentation_logits_resized = self.postprocess_segmentation(segmentation_logits, original_size)
            segmented_image = self.render_segmentation(segmentation_logits_resized, pil_image)
            
            # Convert to ROS message
            result_bgr = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
            result_msg = self.bridge.cv2_to_imgmsg(result_bgr, "bgr8")
            result_msg.header.stamp = self.get_clock().now().to_msg()
            result_msg.header.frame_id = msg.header.frame_id
            
            # Publish result
            self.result_publisher.publish(result_msg)
            
            # Performance logging
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            if len(self.processing_times) > 10:
                self.processing_times.pop(0)
            
            avg_time = np.mean(self.processing_times)
            fps = 1.0 / avg_time if avg_time > 0 else 0
            
            # Log detected objects
            unique_classes = np.unique(segmentation_logits_resized)
            detected_objects = []
            for class_id in unique_classes:
                if class_id > 0:
                    pixel_count = np.sum(segmentation_logits_resized == class_id)
                    percentage = (pixel_count / segmentation_logits_resized.size) * 100
                    if percentage > 0.5:  # Only show significant detections
                        class_name = ADE20K_CLASSES[class_id] if class_id < len(ADE20K_CLASSES) else f"Unknown{class_id}"
                        detected_objects.append(f"{class_name}({class_id}): {percentage:.1f}%")
            
            self.get_logger().info(f"🚀 Processed in {processing_time:.2f}s (avg: {avg_time:.2f}s, {fps:.1f}fps)")
            if detected_objects:
                self.get_logger().info(f"🎯 Detected: {', '.join(detected_objects[:5])}")
            
        except Exception as e:
            self.get_logger().error(f"❌ Processing failed: {str(e)}")
            import traceback
            traceback.print_exc()


def main(args=None):
    rclpy.init(args=args)
    
    node = DINOv2BagSegmentationNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()