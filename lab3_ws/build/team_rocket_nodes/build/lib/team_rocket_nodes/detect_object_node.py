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
        
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )
        
        self._img_subscriber = self.create_subscription(
            CompressedImage,
            '/image_raw/compressed',
            self._image_callback,
            qos_profile)
        
        self.angle_publisher = self.create_publisher(Float32, 'object_angle', 10)
        
        timer_period = 0.1  # 10 Hz
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        self.colormask_lower_limit = np.array([100, 75, 75])
        self.colormask_upper_limit = np.array([120, 200, 200])
        self.image_width = 640
        self.camera_fov = 62.2
        
        self.object_angle = 0.0
        self.object_detected = False
        
        self.get_logger().info('Detect Object node initialized')
    
    def _image_callback(self, msg):
        """Process incoming camera images to detect and track object."""
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return
                
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            height, width = hsv.shape[:2]
            
            self.image_width = width
            
            mask = cv2.inRange(hsv, self.colormask_lower_limit, self.colormask_upper_limit)
            
            mask_img = Image.fromarray(mask)
            bbox = mask_img.getbbox()
            
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                object_x = (x1 + x2) / 2.0
                
                pixel_offset = object_x - (width / 2.0)
                
                angle_per_pixel = math.radians(self.camera_fov) / width
                self.object_angle = pixel_offset * angle_per_pixel
                
                self.object_detected = True
                
            else:
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
