#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from std_msgs.msg import Float32
from geometry_msgs.msg import Point
from sensor_msgs.msg import LaserScan
import math
import numpy as np

class GetObjectRange(Node):
    def __init__(self):
        super().__init__('get_object_range')
        
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )
        
        self.angle_subscriber = self.create_subscription(
            Float32,
            'object_angle',
            self.angle_callback,
            10)
        
        self.lidar_subscriber = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            qos_profile)
        
        self.position_publisher = self.create_publisher(Point, 'object_position', 10)
        
        timer_period = 0.1
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        self.object_angle = 0.0
        self.object_distance = 0.0
        self.lidar_data = None
        self.angle_updated = False
        self.lidar_updated = False
        
        self.lidar_angle_min = 0.0
        self.lidar_angle_max = 0.0
        self.lidar_angle_increment = 0.0
        self.lidar_range_min = 0.0
        self.lidar_range_max = 0.0
        
        self.get_logger().info('Get Object Range node initialized')
    
    def angle_callback(self, msg):
        self.object_angle = msg.data
        self.angle_updated = True
    
    def lidar_callback(self, msg):
        self.lidar_data = msg.ranges
        self.lidar_angle_min = msg.angle_min
        self.lidar_angle_max = msg.angle_max
        self.lidar_angle_increment = msg.angle_increment
        self.lidar_range_min = msg.range_min
        self.lidar_range_max = msg.range_max
        self.lidar_updated = True
    
    def get_lidar_distance_at_angle(self, target_angle):
        if self.lidar_data is None:
            return 0.0
        
        if target_angle < self.lidar_angle_min or target_angle > self.lidar_angle_max:
            target_angle = max(self.lidar_angle_min, min(self.lidar_angle_max, target_angle))
        
        angle_index = int((target_angle - self.lidar_angle_min) / self.lidar_angle_increment)
        angle_index = max(0, min(len(self.lidar_data) - 1, angle_index))
        
        distance = self.lidar_data[angle_index]
        
        if math.isinf(distance) or math.isnan(distance) or distance < self.lidar_range_min or distance > self.lidar_range_max:
            valid_distances = []
            for i in range(max(0, angle_index - 2), min(len(self.lidar_data), angle_index + 3)):
                d = self.lidar_data[i]
                if not (math.isinf(d) or math.isnan(d)) and self.lidar_range_min <= d <= self.lidar_range_max:
                    valid_distances.append(d)
            
            if valid_distances:
                distance = sum(valid_distances) / len(valid_distances)
            else:
                distance = 0.0
        
        return distance
    
    def timer_callback(self):
        if not (self.angle_updated and self.lidar_updated):
            return
        
        self.object_distance = self.get_lidar_distance_at_angle(self.object_angle)
        
        msg = Point()
        msg.x = float(self.object_angle)
        msg.y = float(self.object_distance)
        msg.z = 0.0
        
        self.position_publisher.publish(msg)
        
        angle_deg = math.degrees(self.object_angle)
        self.get_logger().info(f'Object position - Angle: {angle_deg:.1f}°, Distance: {self.object_distance:.2f}m')
        
        self.angle_updated = False

def main(args=None):
    rclpy.init(args=args)
    get_object_range = GetObjectRange()
    
    try:
        rclpy.spin(get_object_range)
    except KeyboardInterrupt:
        pass
    finally:
        get_object_range.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
