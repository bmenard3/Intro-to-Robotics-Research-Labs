#!/usr/bin/env python3
"""
Sign Detector Node - Handles CNN-based sign classification
Subscribes to camera images and publishes detected sign class
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory
import os
import numpy as np
import torch
import cv2
from .models import SmallCNN


class SignDetector(Node):
    def __init__(self):
        super().__init__('sign_detector')
        
        # Declare parameters
        self.declare_parameter('camera_topic', '/image_raw')
        self.declare_parameter('model_name', 'best_model_small.pth')
        self.declare_parameter('debug_mode', True)
        
        # Get parameters
        camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
        model_name = self.get_parameter('model_name').get_parameter_value().string_value
        self.debug_mode = self.get_parameter('debug_mode').get_parameter_value().bool_value
        
        # ========== CNN Model Setup ==========
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        package_share_directory = get_package_share_directory('team_rocket_final')
        model_path = os.path.join(package_share_directory, model_name)
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            model_type = checkpoint.get('model_type', 'small')
            
            self.cnn_model = SmallCNN(num_classes=6)
            self.cnn_model.load_state_dict(checkpoint['model_state_dict'])
            self.cnn_model.to(self.device)
            self.cnn_model.eval()
            
            self.img_size = checkpoint.get('img_size', 64)
            self.class_names = ['empty', 'left', 'right', 'do_not_enter', 'stop', 'goal']
            
            self.get_logger().info('='*60)
            self.get_logger().info(f'✓ Sign Detector Initialized')
            self.get_logger().info(f'  Model: {model_type.upper()} CNN')
            self.get_logger().info(f'  Accuracy: {checkpoint["val_acc"]:.2f}%')
            self.get_logger().info(f'  Device: {self.device}')
            self.get_logger().info(f'  Image size: {self.img_size}x{self.img_size}')
            self.get_logger().info('='*60)
        except Exception as e:
            self.get_logger().error(f'Failed to load CNN model: {e}')
            raise
        
        # ========== CV Bridge ==========
        self.bridge = CvBridge()
        self.latest_image = None
        self.image_count = 0
        
        # ========== ROS2 Subscribers ==========
        self.camera_subscriber = self.create_subscription(
            Image,
            camera_topic,
            self.camera_callback,
            10
        )
        self.get_logger().info(f'📷 Subscribed to camera: {camera_topic}')
        
        # ========== ROS2 Publishers ==========
        self.detection_publisher = self.create_publisher(
            Int32,
            '/sign_detection',
            10
        )
        self.get_logger().info(f'📢 Publishing detections to: /sign_detection')
        
        # ========== Timer for periodic detection ==========
        # Detection runs at 5Hz
        timer_period = 0.2  # 5Hz
        self.timer = self.create_timer(timer_period, self.detect_and_publish)
        self.get_logger().info(f'🔄 Auto-detection enabled at 5Hz')
    
    def camera_callback(self, msg):
        """Callback for camera images"""
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.image_count += 1
            
            # Log every 50 frames to avoid spam
            if self.image_count % 50 == 0:
                self.get_logger().info(
                    f'📷 Camera OK: {self.image_count} images, shape={self.latest_image.shape}'
                )
        except Exception as e:
            self.get_logger().error(f'❌ Image conversion failed: {e}')
    
    def preprocess_image(self, image):
        """
        Preprocess image for CNN inference
        Args:
            image: BGR image from camera (numpy array)
        Returns:
            tensor: Preprocessed image tensor ready for model
        """
        if image is None:
            self.get_logger().warn('⚠️ preprocess_image: input is None')
            return None
        
        try:
            # Resize to model input size
            image = cv2.resize(image, (self.img_size, self.img_size))
            
            # BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Normalize to [0, 1]
            image = image.astype(np.float32) / 255.0
            
            # Convert to (C, H, W) format
            image = np.transpose(image, (2, 0, 1))
            
            # Convert to tensor and add batch dimension
            tensor = torch.FloatTensor(image).unsqueeze(0)
            
            return tensor
        except Exception as e:
            self.get_logger().error(f'❌ preprocess_image error: {e}')
            return None
    
    def detect_sign(self):
        """
        Run sign detection on latest camera image
        Returns:
            int: Detected class (0-5) or -1 if error/no image
                 0=empty, 1=left, 2=right, 3=do_not_enter, 4=stop, 5=goal
        """
        # Check if camera image is available
        if self.latest_image is None:
            return -1
        
        # Preprocess image
        input_tensor = self.preprocess_image(self.latest_image)
        if input_tensor is None:
            self.get_logger().error('❌ Preprocessing failed')
            return -1
        
        # Move to device
        input_tensor = input_tensor.to(self.device)
        
        # Run inference
        try:
            with torch.no_grad():
                outputs = self.cnn_model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                pred_class = predicted.item()
                conf_value = confidence.item()
        except Exception as e:
            self.get_logger().error(f'❌ CNN inference error: {e}')
            return -1
        
        # Log results
        class_name = self.class_names[pred_class]
        
        if self.debug_mode:
            self.get_logger().info(f'🔍 CNN Detection:')
            self.get_logger().info(f'   Class: {pred_class} ({class_name})')
            self.get_logger().info(f'   Confidence: {conf_value:.3f}')
            
            # Print all class probabilities
            probs = probabilities[0].cpu().numpy()
            prob_str = ', '.join([
                f'{i}({self.class_names[i]}):{probs[i]:.2f}' 
                for i in range(6)
            ])
            self.get_logger().info(f'   Probs: [{prob_str}]')
        else:
            # Compact logging in auto mode
            self.get_logger().info(
                f'🔍 {class_name.upper()} (conf: {conf_value:.2f})'
            )
        
        return pred_class
    
    def detect_and_publish(self):
        """Detect sign and publish result"""
        detection = self.detect_sign()
        
        msg = Int32()
        msg.data = detection
        self.detection_publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    sign_detector = SignDetector()
    
    try:
        rclpy.spin(sign_detector)
    except KeyboardInterrupt:
        pass
    finally:
        sign_detector.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
