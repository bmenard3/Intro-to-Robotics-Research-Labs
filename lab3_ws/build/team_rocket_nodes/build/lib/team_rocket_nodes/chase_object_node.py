#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from geometry_msgs.msg import Point, Twist
import math
import time

class PIDController:
    def __init__(self, kp=1.0, ki=0.0, kd=0.0, integral_limit=1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self.prev_error = 0.0
        self.integral = 0.0
        self.prev_time = time.time()
    
    def compute(self, error, current_time=None):
        if current_time is None:
            current_time = time.time()
        
        dt = current_time - self.prev_time
        if dt <= 0.0:
            dt = 0.01
        
        proportional = self.kp * error
        
        self.integral += error * dt
        self.integral = max(-self.integral_limit, min(self.integral_limit, self.integral))
        integral_term = self.ki * self.integral
        
        derivative = (error - self.prev_error) / dt
        derivative_term = self.kd * derivative
        
        output = proportional + integral_term + derivative_term
        
        self.prev_error = error
        self.prev_time = current_time
        
        return output
    
    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0
        self.prev_time = time.time()

class ChaseObject(Node):
    def __init__(self):
        super().__init__('chase_object')
        
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )
        
        self.position_subscriber = self.create_subscription(
            Point,
            'object_position',
            self.position_callback,
            10)
        
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        
        timer_period = 0.1
        self.timer = self.create_timer(timer_period, self.control_callback)
        
        self.desired_distance = 0.5
        self.distance_tolerance = 0.05
        self.angle_tolerance = math.radians(2)
        
        self.object_angle = 0.0
        self.object_distance = 0.0
        self.position_received = False
        
        self.angular_pid = PIDController(
            kp=4.0,
            ki=1.2,
            kd=1.0,
            integral_limit=0.5
        )
        
        self.linear_pid = PIDController(
            kp=1.0,
            ki=0.0,
            kd=0.0,
            integral_limit=0.3
        )
        
        self.max_linear_speed = 0.3
        self.max_angular_speed = 1.0
        self.min_safe_distance = 0.15
        
        self.get_logger().info('Chase Object node initialized')
    
    def position_callback(self, msg):
        self.object_angle = msg.x
        self.object_distance = msg.y
        self.position_received = True
    
    def control_callback(self):
        if not self.position_received:
            self.publish_zero_velocity()
            return
        
        if self.object_distance < self.min_safe_distance and self.object_distance > 0.0:
            self.get_logger().warn(f'Object too close ({self.object_distance:.2f}m)! Stopping.')
            self.publish_zero_velocity()
            return
        
        angular_error = -self.object_angle
        distance_error = self.object_distance - self.desired_distance
        
        current_time = time.time()
        
        angular_velocity = self.angular_pid.compute(angular_error, current_time)
        
        if abs(self.object_angle) < math.radians(15):
            linear_velocity = self.linear_pid.compute(distance_error, current_time)
        else:
            linear_velocity = 0.0
            self.linear_pid.reset()
        
        angular_velocity = max(-self.max_angular_speed, min(self.max_angular_speed, angular_velocity))
        linear_velocity = max(-self.max_linear_speed, min(self.max_linear_speed, linear_velocity))
        
        if self.object_distance <= 0.0 or self.object_distance > 5.0:
            linear_velocity = 0.0
            self.linear_pid.reset()
        
        cmd_msg = Twist()
        cmd_msg.linear.x = linear_velocity
        cmd_msg.linear.y = 0.0
        cmd_msg.linear.z = 0.0
        cmd_msg.angular.x = 0.0
        cmd_msg.angular.y = 0.0
        cmd_msg.angular.z = angular_velocity
        
        self.cmd_vel_publisher.publish(cmd_msg)
        
        angle_deg = math.degrees(self.object_angle)
        ang_vel_deg = math.degrees(angular_velocity)
        
        control_mode = "ANGULAR ONLY" if linear_velocity == 0.0 else "ANGULAR + LINEAR"
        
        self.get_logger().info(
            f'Control [{control_mode}]: Angle={angle_deg:.1f}°, Dist={self.object_distance:.2f}m, '
            f'AngVel={ang_vel_deg:.1f}°/s, LinVel={linear_velocity:.2f}m/s'
        )
        
        if (abs(self.object_angle) < self.angle_tolerance and 
            abs(distance_error) < self.distance_tolerance and 
            self.object_distance > 0.0):
            self.get_logger().info('TARGET REACHED - Object tracking successful!')
    
    def publish_zero_velocity(self):
        cmd_msg = Twist()
        cmd_msg.linear.x = 0.0
        cmd_msg.linear.y = 0.0
        cmd_msg.linear.z = 0.0
        cmd_msg.angular.x = 0.0
        cmd_msg.angular.y = 0.0
        cmd_msg.angular.z = 0.0
        self.cmd_vel_publisher.publish(cmd_msg)

def main(args=None):
    rclpy.init(args=args)
    chase_object = ChaseObject()
    
    try:
        rclpy.spin(chase_object)
    except KeyboardInterrupt:
        pass
    finally:
        chase_object.publish_zero_velocity()
        chase_object.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
