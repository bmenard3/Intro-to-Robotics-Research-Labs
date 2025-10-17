# goToGoal_modified.py

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Point
from std_msgs.msg import Bool       # Modified import
from sensor_msgs.msg import LaserScan # Modified import
import numpy as np
import math
import time

# ... (Keep your PIDController class exactly as it is) ...
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

class GoToGoal(Node):
    def __init__(self):
        super().__init__('go_to_goal')
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth = 1
        )
        # --- MODIFIED SUBSCRIBERS ---
        self.scan_subscriber = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.odom_subscriber = self.create_subscription(Odometry, '/odom', self.odom_callback, qos_profile)
        self.avoider_feedback_subscriber = self.create_subscription(Bool, '/avoidance_active', self.avoider_feedback_callback, 10)
        
        # --- MODIFIED PUBLISHERS ---
        self.vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.goal_publisher = self.create_publisher(Point, '/current_goal', 10)
        self.avoider_active_publisher = self.create_publisher(Bool, '/avoidance_active', 10)

        # --- TIMER AND STATE ---
        timer_period = 0.1
        self.timer = self.create_timer(timer_period, self.navigation_loop)
        self.state = 0 # 0: GO_TO_GOAL, 1: AVOIDING, 2: WAIT_AT_GOAL
        self.wait_start = 0
        self.obstacle_in_path = False
        self.obstacle_threshold = 0.35 # meters
        self.scan_data = None
        
        # --- ODOMETRY AND WAYPOINTS ---
        self.Init = True
        self.Init_pos = Point()
        self.Init_pos.x = 0.0
        self.Init_pos.y = 0.0
        self.Init_ang = 0.0
        self.globalPos = Point()
        self.globalAng = 0.0
        
        self.waypoints = np.array([[1.5, 0], [1.5, 1.4], [0, 1.4]])
        self.current_goal = 0
        self.goal_pos = Point()

        # --- CONTROLLERS ---
        self.angular_pid = PIDController(kp=0.8, ki=0.0, kd=0.0, integral_limit=3.0)
        self.max_angular_speed = 1.0
        self.max_linear_speed = 0.20

        self.get_logger().info(f'GoToGoal Master Node Initialized.')
        
    def odom_callback(self, msg):
        self.update_Odometry(msg)

    def scan_callback(self, msg):
        self.scan_data = msg
        # Simple frontal obstacle check
        num_ranges = len(msg.ranges)
        center_index = num_ranges // 2
        cone_width_indices = 15 # Check +/- 15 degrees
        
        start_index = max(0, center_index - cone_width_indices)
        end_index = min(num_ranges - 1, center_index + cone_width_indices)
        
        frontal_ranges = [r for r in msg.ranges[start_index:end_index] if np.isfinite(r)]
        if not frontal_ranges:
            self.obstacle_in_path = False
            return
            
        if min(frontal_ranges) < self.obstacle_threshold:
            self.obstacle_in_path = True
        else:
            self.obstacle_in_path = False

    def avoider_feedback_callback(self, msg):
        # When the avoider says it's done (sends False), we can go back to goal seeking
        if not msg.data and self.state == 1: # State 1 is AVOIDING
            self.get_logger().info("Avoider has completed its task. Resuming GoToGoal.")
            self.state = 0

    def navigation_loop(self):
        # State machine
        if self.state == 0: # Go To Goal
            # Check for obstacles BEFORE moving
            if self.obstacle_in_path:
                self.get_logger().info("Obstacle detected! Handing control to Avoider.")
                self.stop_robot()
                self.state = 1 # Switch to AVOIDING state
                
                # Activate the avoider node
                active_msg = Bool()
                active_msg.data = True
                self.avoider_active_publisher.publish(active_msg)
                return # Exit this loop cycle
            
            self.perform_go_to_goal()

        elif self.state == 1: # Avoid Obstacles (Master is just waiting)
            # Publish the current goal so the avoider knows where to go
            goal_msg = Point()
            goal_msg.x = self.waypoints[self.current_goal, 0]
            goal_msg.y = self.waypoints[self.current_goal, 1]
            self.goal_publisher.publish(goal_msg)
            # The master node does nothing else while the slave is active

        elif self.state == 2: # Wait At Goal
            self.perform_wait_at_goal()

    def perform_go_to_goal(self):
        self.goal_pos.x = self.waypoints[self.current_goal, 0]
        self.goal_pos.y = self.waypoints[self.current_goal, 1]
        dx = self.goal_pos.x - self.globalPos.x
        dy = self.goal_pos.y - self.globalPos.y
        d = math.sqrt(dx**2 + dy**2)
        
        target_angle = math.atan2(dy, dx)
        d_theta = self.normalize_angle(target_angle - self.globalAng)

        msg = Twist()
        if d > 0.10: # Goal radius
            if abs(d_theta) > 0.2: # If angle error is large, just turn
                msg.linear.x = 0.0
            else:
                msg.linear.x = min(self.max_linear_speed, d * 0.5) # P-control on linear speed
            
            angular_velocity = self.angular_pid.compute(d_theta, time.time())
            msg.angular.z = np.clip(angular_velocity, -self.max_angular_speed, self.max_angular_speed)
        else:
            self.get_logger().info(f"Reached Waypoint {self.current_goal}")
            self.stop_robot()
            self.state = 2
            self.wait_start = time.time()
        
        self.vel_publisher.publish(msg)

    def perform_wait_at_goal(self):
        self.stop_robot()
        wait_duration = 10 if self.current_goal == 0 else 2 # 10s for first goal, 2s otherwise
        
        if (time.time() - self.wait_start) > wait_duration:
            self.current_goal += 1
            if self.current_goal >= self.waypoints.shape[0]:
                self.get_logger().info('All waypoints reached. Completed!')
                self.timer.cancel() # Stop the navigation loop
                self.stop_robot()
            else:
                self.get_logger().info(f"Proceeding to next waypoint.")
                self.state = 0

    def stop_robot(self):
        self.vel_publisher.publish(Twist())

    # ... (Keep your update_Odometry and normalize_angle methods) ...
    def update_Odometry(self, Odom):
        position = Odom.pose.pose.position
        q = Odom.pose.pose.orientation
        orientation = np.arctan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))

        if self.Init:
            self.Init = False
            self.Init_ang = orientation
            self.globalAng = self.Init_ang
            Mrot = np.matrix([[np.cos(self.Init_ang), np.sin(self.Init_ang)],[-np.sin(self.Init_ang), np.cos(self.Init_ang)]])        
            self.Init_pos.x = Mrot.item((0,0))*position.x + Mrot.item((0,1))*position.y
            self.Init_pos.y = Mrot.item((1,0))*position.x + Mrot.item((1,1))*position.y
            self.Init_pos.z = position.z
        
        Mrot = np.matrix([[np.cos(self.Init_ang), np.sin(self.Init_ang)],[-np.sin(self.Init_ang), np.cos(self.Init_ang)]])        
        self.globalPos.x = Mrot.item((0,0))*position.x + Mrot.item((0,1))*position.y - self.Init_pos.x
        self.globalPos.y = Mrot.item((1,0))*position.x + Mrot.item((1,1))*position.y - self.Init_pos.y
        globalAng = orientation - self.Init_ang
        self.globalAng = self.normalize_angle(globalAng)

    @staticmethod
    def normalize_angle(angle):
        while angle > math.pi: angle -= 2.0 * math.pi
        while angle < -math.pi: angle += 2.0 * math.pi
        return angle

def main(args=None):
    rclpy.init(args=args)
    go_to_goal = GoToGoal()
    rclpy.spin(go_to_goal)
    go_to_goal.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()