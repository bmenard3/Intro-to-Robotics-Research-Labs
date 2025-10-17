# obstacle_avoider.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
import numpy as np
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
            dt = 1e-6 # Prevent division by zero
        
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

class ObstacleAvoider(Node):
    def __init__(self):
        super().__init__('obstacle_avoider')
        
        # --- Parameters ---
        self.declare_parameter('follow_distance', 0.4)       # Target distance from the wall (meters)
        self.declare_parameter('follow_speed', 0.1)          # Linear speed while following (m/s)
        self.declare_parameter('max_angular_speed', 0.7)     # Max turning speed (rad/s)

        # --- State Variables ---
        self.active = False
        self.scan_data = None
        self.current_goal = None
        self.current_pos = None
        self.current_theta = None
        
        # PID to maintain distance from the wall. Error is (desired_distance - current_distance)
        self.distance_pid = PIDController(kp=1.0, ki=0.0, kd=0.0, integral_limit=1.0)

        # --- Publishers and Subscribers ---
        self.vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.scan_subscriber = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.goal_subscriber = self.create_subscription(Point, '/current_goal', self.goal_callback, 10)
        self.odom_subscriber = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.active_subscriber = self.create_subscription(Bool, '/avoidance_active', self.active_callback, 10)
        
        # Publisher to signal when avoidance is done
        self.feedback_publisher = self.create_publisher(Bool, '/avoidance_active', 10)
        
        # --- Control Loop ---
        self.timer = self.create_timer(0.1, self.control_loop) # 10 Hz
        self.get_logger().info("Obstacle Avoider node is ready and waiting for activation.")

    def active_callback(self, msg):
        if msg.data and not self.active:
            self.get_logger().info("--- Activating Obstacle Avoidance ---")
            self.active = True
        elif not msg.data and self.active:
            self.get_logger().info("--- Deactivating Obstacle Avoidance ---")
            self.active = False

    def scan_callback(self, msg):
        self.scan_data = msg

    def goal_callback(self, msg):
        self.current_goal = msg

    def odom_callback(self, msg):
        self.current_pos = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.current_theta = np.arctan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))

    def control_loop(self):
        # Only run logic if the node is active and we have the necessary data
        if not self.active or self.scan_data is None or self.current_goal is None or self.current_pos is None:
            return

        # Check if the path to the goal is clear
        if self.is_path_to_goal_clear():
            self.get_logger().info("Path to goal is clear. Handing back control.")
            self.stop_robot()
            # Publish False to deactivate self and reactivate the goToGoal node
            feedback_msg = Bool()
            feedback_msg.data = False
            self.feedback_publisher.publish(feedback_msg)
            self.active = False
            return

        # If path is not clear, perform wall-following logic
        self.wall_follow_logic()

    def wall_follow_logic(self):
        # We will follow the wall on the robot's right side
        # Extract the lidar readings from the side (e.g., -100 to -80 degrees for right side)
        angle_min_deg = -100
        angle_max_deg = -80
        
        # Convert degrees to indices
        angle_min_rad = np.deg2rad(angle_min_deg)
        angle_max_rad = np.deg2rad(angle_max_deg)
        
        start_index = int((angle_min_rad - self.scan_data.angle_min) / self.scan_data.angle_increment)
        end_index = int((angle_max_rad - self.scan_data.angle_min) / self.scan_data.angle_increment)
        
        # Get a clean array of distances on the right side
        side_ranges = [r for r in self.scan_data.ranges[start_index:end_index] if np.isfinite(r)]
        
        if not side_ranges:
            # Lost the wall, turn right to find it again
            self.get_logger().warn("Lost the wall, turning to find it.")
            twist_msg = Twist()
            twist_msg.linear.x = 0.05 
            twist_msg.angular.z = -0.5 # Turn right
            self.vel_publisher.publish(twist_msg)
            return

        current_distance = np.mean(side_ranges)
        desired_distance = self.get_parameter('follow_distance').get_parameter_value().double_value
        
        error = desired_distance - current_distance
        
        # Use PID to calculate the required angular velocity to maintain distance
        angular_velocity = self.distance_pid.compute(error, time.time())
        
        # Get parameters
        follow_speed = self.get_parameter('follow_speed').get_parameter_value().double_value
        max_angular = self.get_parameter('max_angular_speed').get_parameter_value().double_value
        
        twist_msg = Twist()
        twist_msg.linear.x = follow_speed
        twist_msg.angular.z = np.clip(angular_velocity, -max_angular, max_angular)
        
        self.vel_publisher.publish(twist_msg)

    def is_path_to_goal_clear(self):
        # Calculate the direct angle from robot to goal
        dx = self.current_goal.x - self.current_pos.x
        dy = self.current_goal.y - self.current_pos.y
        angle_to_goal = math.atan2(dy, dx)
        
        # Find the index in the laser scan that corresponds to this angle
        # This requires converting from the global frame angle to the robot's local frame angle
        relative_angle_to_goal = self.normalize_angle(angle_to_goal - self.current_theta)
        goal_index = int((relative_angle_to_goal - self.scan_data.angle_min) / self.scan_data.angle_increment)
        
        # Check a small cone around the direct path to the goal
        cone_width_indices = 5 # Check +/- 5 indices
        start_index = max(0, goal_index - cone_width_indices)
        end_index = min(len(self.scan_data.ranges) - 1, goal_index + cone_width_indices)
        
        distance_to_goal = math.sqrt(dx**2 + dy**2)
        
        for i in range(start_index, end_index + 1):
            # If any laser reading on the path to the goal is finite and SHORTER 
            # than the actual distance to the goal, the path is blocked.
            if np.isfinite(self.scan_data.ranges[i]) and self.scan_data.ranges[i] < (distance_to_goal * 1.1): # 10% buffer
                return False
                
        return True

    def stop_robot(self):
        self.vel_publisher.publish(Twist())

    @staticmethod
    def normalize_angle(angle):
        while angle > math.pi: angle -= 2.0 * math.pi
        while angle < -math.pi: angle += 2.0 * math.pi
        return angle

def main(args=None):
    rclpy.init(args=args)
    avoider_node = ObstacleAvoider()
    rclpy.spin(avoider_node)
    avoider_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()