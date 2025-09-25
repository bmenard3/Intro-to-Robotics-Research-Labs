#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from std_msgs.msg import Float32
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np
from PIL import Image
import math

class DetectObject(Node):
    def __init__(self):
        super().__init__('detect_object')
        
        # QoS profile for camera subscription
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )
        
        # Subscriber to camera feed
        self._img_subscriber = self.create_subscription(
            CompressedImage,
            '/image_raw/compressed',
            self._image_callback,
            qos_profile)
        
        # Publisher for object angular position
        self.angle_publisher = self.create_publisher(Float32, 'object_angle', 10)
        
        # Timer for publishing at regular intervals
        timer_period = 0.1  # 10 Hz
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        # Color mask parameters (HSV) - adjust these for your target object
        self.colormask_lower_limit = np.array([100, 75, 75])   # Lower blue range
        self.colormask_upper_limit = np.array([120, 200, 200]) # Upper blue range
        
        # Camera parameters
        self.image_width = 640  # Typical camera resolution
        self.camera_fov = 62.2  # TurtleBot3 camera field of view in degrees
        
        # Object tracking variables
        self.object_angle = 0.0  # Angular position in radians
        self.object_detected = False
        
        self.get_logger().info('Detect Object node initialized')
    
    def _image_callback(self, msg):
        """Process incoming camera images to detect and track object."""
        try:
            # Convert compressed image to OpenCV format
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return
                
            # Convert to HSV for color filtering
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            height, width = hsv.shape[:2]
            
            # Update image width for angle calculations
            self.image_width = width
            
            # Create color mask
            mask = cv2.inRange(hsv, self.colormask_lower_limit, self.colormask_upper_limit)
            
            # Find object bounding box
            mask_img = Image.fromarray(mask)
            bbox = mask_img.getbbox()
            
            if bbox is not None:
                # Object detected - calculate center position
                x1, y1, x2, y2 = bbox
                object_x = (x1 + x2) / 2.0
                
                # Convert pixel position to angular position
                # Pixel offset from image center
                pixel_offset = object_x - (width / 2.0)
                
                # Convert to angle in radians
                # Angle per pixel = field_of_view / image_width
                angle_per_pixel = math.radians(self.camera_fov) / width
                self.object_angle = pixel_offset * angle_per_pixel
                
                self.object_detected = True
                
            else:
                # No object detected
                self.object_detected = False
                self.object_angle = 0.0
                
        except Exception as e:
            self.get_logger().error(f'Image processing error: {str(e)}')
    
    def timer_callback(self):
        """Publish object angular position at regular intervals."""
        msg = Float32()
        msg.data = self.object_angle
        self.angle_publisher.publish(msg)
        
        if self.object_detected:
            self.get_logger().info(f'Object detected at angle: {math.degrees(self.object_angle):.2f} degrees')
        else:
            self.get_logger().debug('No object detected')

def main(args=None):
    rclpy.init(args=args)
    detect_object = DetectObject()
    
    try:
        rclpy.spin(detect_object)
    except KeyboardInterrupt:
        pass
    finally:
        detect_object.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
