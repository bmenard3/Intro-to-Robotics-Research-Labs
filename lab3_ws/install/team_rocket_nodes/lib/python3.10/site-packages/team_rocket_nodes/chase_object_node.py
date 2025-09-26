#!/usr/bin/env python3
"""
chase_object.py
Authors: [Your Names Here]
Date: September 2025

ROS2 node that implements dual PID controllers to chase an object.
Controls both angular rotation (to face object) and linear movement (to maintain distance).
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from geometry_msgs.msg import Point, Twist
import math
import time

class PIDController:
    """Simple PID controller implementation."""
    
    def __init__(self, kp=1.0, ki=0.0, kd=0.0, integral_limit=1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        
        self.prev_error = 0.0
        self.integral = 0.0
        self.prev_time = time.time()
    
    def compute(self, error, current_time=None):
        """Compute PID output given current error."""
        if current_time is None:
            current_time = time.time()
        
        dt = current_time - self.prev_time
        if dt <= 0.0:
            dt = 0.01  # Prevent division by zero
        
        # Proportional term
        proportional = self.kp * error
        
        # Integral term with windup protection
        self.integral += error * dt
        # Clamp integral to prevent windup
        self.integral = max(-self.integral_limit, min(self.integral_limit, self.integral))
        integral_term = self.ki * self.integral
        
        # Derivative term
        derivative = (error - self.prev_error) / dt
        derivative_term = self.kd * derivative
        
        # Compute total output
        output = proportional + integral_term + derivative_term
        
        # Update for next iteration
        self.prev_error = error
        self.prev_time = current_time
        
        return output
    
    def reset(self):
        """Reset PID controller state."""
        self.prev_error = 0.0
        self.integral = 0.0
        self.prev_time = time.time()

class ChaseObject(Node):
    def __init__(self):
        super().__init__('chase_object')
        
        # QoS profile
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )
        
        # Subscriber to object position
        self.position_subscriber = self.create_subscription(
            Point,
            'object_position',
            self.position_callback,
            10)
        
        # Publisher for robot velocity commands
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Timer for control loop
        timer_period = 0.1  # 10 Hz control loop
        self.timer = self.create_timer(timer_period, self.control_callback)
        
        # Control parameters
        self.desired_distance = 0.5  # Desired distance from object (meters)
        self.distance_tolerance = 0.05  # Tolerance for distance control
        self.angle_tolerance = math.radians(2)  # Tolerance for angular control (2 degrees)
        
        # Current object state
        self.object_angle = 0.0  # Current object angle (radians)
        self.object_distance = 0.0  # Current object distance (meters)
        self.position_received = False
        
        # PID Controllers
        # Angular controller: controls rotation to face object
        self.angular_pid = PIDController(
            kp=3.5,    # Proportional gain
            ki=0.0,
            kd=0.0,
            #kp=3.8
            #ki=1.5,    # Integral gain (small to prevent windup)
            #kd=0.5,    # Derivative gain (helps with stability)
            integral_limit=0.5  # Limit integral windup
        )
        
        # Linear controller: controls forward/backward motion to maintain distance
        self.linear_pid = PIDController(
            kp=0.5,    # Proportional gain
            ki=0.0,   # Small integral gain
            kd=0.0,    # Derivative gain
            integral_limit=0.3  # Limit integral windup
        )
        
        # Safety limits
        self.max_linear_speed = 0.3  # m/s
        self.max_angular_speed = 1.0  # rad/s
        self.min_safe_distance = 0.15  # Minimum safe distance (meters)
        
        self.get_logger().info('Chase Object node initialized')
        self.get_logger().info(f'Desired distance: {self.desired_distance}m, Min safe distance: {self.min_safe_distance}m')
    
    def position_callback(self, msg):
        """Receive object position from get_object_range node."""
        self.object_angle = msg.x  # Angular position (radians)
        if self.object_angle == 0.0:
            self.object_distance = 0.0
            self.position_received = False
        else:
            self.object_distance = msg.y  # Linear distance (meters)
            self.position_received = True
    
    def control_callback(self):
        """Main control loop - compute and publish velocity commands."""
        if not self.position_received:
            # No object position received yet, stop robot
            self.publish_zero_velocity()
            return
        
        # Check if object is too close (safety stop)
        if self.object_distance < self.min_safe_distance and self.object_distance > 0.0:
            self.get_logger().warn(f'Object too close ({self.object_distance:.2f}m)! Stopping.')
            self.publish_zero_velocity()
            return
        
        # Compute control errors
        angular_error = -self.object_angle  # Negative because we want to turn toward object
        distance_error = self.object_distance - self.desired_distance  # Positive = too far, negative = too close
        
        current_time = time.time()
        
        # Compute PID outputs
        angular_velocity = self.angular_pid.compute(angular_error, current_time)
        linear_velocity = self.linear_pid.compute(distance_error, current_time)
        
        # Apply safety limits
        angular_velocity = max(-self.max_angular_speed, min(self.max_angular_speed, angular_velocity))
        linear_velocity = max(-self.max_linear_speed, min(self.max_linear_speed, linear_velocity))
        
        # If object distance is invalid (0 or very large), only do angular control
        if self.object_distance <= 0.0 or self.object_distance > 5.0:
            linear_velocity = 0.0
            self.linear_pid.reset()  # Reset linear PID when no valid distance
        
        # Create and publish velocity command
        cmd_msg = Twist()
        cmd_msg.linear.x = linear_velocity
        cmd_msg.linear.y = 0.0
        cmd_msg.linear.z = 0.0
        cmd_msg.angular.x = 0.0
        cmd_msg.angular.y = 0.0
        cmd_msg.angular.z = angular_velocity
        
        self.cmd_vel_publisher.publish(cmd_msg)
        
        # Log control information
        angle_deg = math.degrees(self.object_angle)
        ang_vel_deg = math.degrees(angular_velocity)
        
        self.get_logger().info(
            f'Control: Angle={angle_deg:.1f}°, Dist={self.object_distance:.2f}m, '
            f'AngVel={ang_vel_deg:.1f}°/s, LinVel={linear_velocity:.2f}m/s'
        )
        
        # Check if robot has reached target (within tolerance)
        if (abs(self.object_angle) < self.angle_tolerance and  # Use abs(object_angle) instead of abs(angular_error)
            abs(distance_error) < self.distance_tolerance and 
            self.object_distance > 0.0):
            self.get_logger().info('TARGET REACHED - Object tracking successful!')
    
    def publish_zero_velocity(self):
        """Publish zero velocity to stop the robot."""
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
        # Ensure robot stops when node shuts down
        chase_object.publish_zero_velocity()
        chase_object.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
