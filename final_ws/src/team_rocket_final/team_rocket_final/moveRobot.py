#!/usr/bin/env python3
"""
Motion Controller Node - Handles robot navigation and state machine
Subscribes to laser scans, odometry, and sign detections
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Point
from std_msgs.msg import Float32MultiArray, Int32
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


class MoveRobot(Node):
    def __init__(self):
        super().__init__('move_robot')
        
        # QoS Profile for sensors
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )
        
        # ========== Sign Detection State ==========
        self.latest_detection = -1  # -1 = no detection yet
        self.detection_count = 0
        self.class_names = ['empty', 'left', 'right', 'do_not_enter', 'stop', 'goal']
        
        # ========== ROS2 Subscribers ==========
        self.range_subscriber = self.create_subscription(
            Float32MultiArray,
            '/ranges',
            self.range_callback,
            qos_profile
        )
        
        self.angle_subscriber = self.create_subscription(
            Float32MultiArray,
            '/angles',
            self.angle_callback,
            qos_profile
        )
        
        self.scan_subscriber = self.create_subscription(
            Float32MultiArray,
            '/processed_scans',
            self.scan_callback,
            qos_profile
        )
        
        self.odom_subscriber = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            qos_profile
        )
        
        # Subscribe to sign detections
        self.detection_subscriber = self.create_subscription(
            Int32,
            '/sign_detection',
            self.detection_callback,
            10
        )
        
        # ========== ROS2 Publishers ==========
        self.vel_publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )
        
        # ========== Navigation Timer ==========
        timer_period = 0.1  # 10Hz
        self.timer = self.create_timer(timer_period, self.navigation)
        
        # ========== PID Controller ==========
        self.angular_pid = PIDController(
            kp=0.5,
            ki=0.0,
            kd=0.0,
            integral_limit=3.0
        )
        self.max_angular_speed = 1.0
        
        # ========== Odometry ==========
        self.movement_start = False
        self.Init_pos = Point()
        self.Init_pos.x = 0.0
        self.Init_pos.y = 0.0
        self.Init_ang = 0.0
        self.globalPos = Point()
        self.globalAng = 0.0
        
        # ========== State Machine ==========
        self.state = 1  # Start in forward state
        
        # ========== Sensor Data ==========
        self.ranges = [0.5]
        self.angles = [math.pi]
        self.forward_distance = 1.0
        self.right_distance = 2.0
        self.left_distance = 2.0
        
        # ========== Small Turn Angle ==========
        self.small_turn_angle = math.pi / 6  # 30 degrees for obstacle avoidance
        
        self.get_logger().info('='*60)
        self.get_logger().info('✓ Motion Controller Initialized')
        self.get_logger().info('  Subscribed to: /ranges, /angles, /processed_scans, /odom, /sign_detection')
        self.get_logger().info('  Publishing to: /cmd_vel')
        self.get_logger().info('  Navigation frequency: 10Hz')
        self.get_logger().info('='*60)
    
    def detection_callback(self, msg):
        """Callback for sign detection messages"""
        self.latest_detection = msg.data
        self.detection_count += 1
        
        if self.latest_detection >= 0 and self.latest_detection <= 5:
            class_name = self.class_names[self.latest_detection]
            self.get_logger().info(
                f'📥 Received detection: {self.latest_detection} ({class_name})'
            )
    
    def odom_callback(self, msg):
        """Callback for odometry"""
        self.update_Odometry(msg)
    
    def range_callback(self, msg):
        """Callback for range data"""
        self.ranges = msg.data
    
    def angle_callback(self, msg):
        """Callback for angle data"""
        angles = np.array(msg.data)
        self.angles = list((angles + np.pi) % (2 * np.pi) - np.pi)
    
    def scan_callback(self, msg):
        """Callback for processed laser scans"""
        N = msg.layout.dim[0].size
        M = msg.layout.dim[1].size
        self.latest_scan = np.array(msg.data, dtype=np.float32).reshape(N, M)
        
        # Replace NaN values with max range
        for i in range(self.latest_scan.shape[0]):
            if np.isnan(self.latest_scan[i, 1]):
                self.latest_scan[i, 1] = 5.0
        
        # Extract forward scan (±15 degrees)
        forward_scan_ind0 = np.argmin(np.abs(self.latest_scan[:, 0] + (math.pi / 12)))
        forward_scan_ind1 = np.argmin(np.abs(self.latest_scan[:, 0] - (math.pi / 12)))
        self.forward_scan = self.latest_scan[forward_scan_ind0:forward_scan_ind1, :]
        self.forward_distance = np.min(self.forward_scan[:, 1])
        
        # Extract right scan (90° ± 30°)
        right_scan_ind0 = np.argmin(np.abs(self.latest_scan[:, 0] + ((math.pi / 2) + (math.pi / 8))))
        right_scan_ind1 = np.argmin(np.abs(self.latest_scan[:, 0] + ((math.pi / 2) - (math.pi / 8))))
        self.right_scan = self.latest_scan[right_scan_ind0:right_scan_ind1, :]
        self.right_distance = np.min(self.right_scan[:, 1])
        
        # Extract left scan (90° ± 30°)
        left_scan_ind0 = np.argmin(np.abs(self.latest_scan[:, 0] - ((math.pi / 2) - (math.pi / 8))))
        left_scan_ind1 = np.argmin(np.abs(self.latest_scan[:, 0] - ((math.pi / 2) + (math.pi / 8))))
        self.left_scan = self.latest_scan[left_scan_ind0:left_scan_ind1, :]
        self.left_distance = np.min(self.left_scan[:, 1])
    
    def get_latest_detection(self):
        """
        Get the latest sign detection
        Returns:
            int: Detection class (0-5) or -1 if no detection available
        """
        return self.latest_detection
    
    def navigation(self):
        """Main navigation state machine"""
        msg = Twist()
        closest_scan = self.ranges.index(min(self.ranges))
        closest_range = self.ranges[closest_scan]
        closest_angle = self.angles[closest_scan]
        
        # ========== State 0: Detect Sign ==========
        if self.state == 0:
            self.get_logger().info('State 0: Detecting sign...')
            if (closest_range < 0.5) and (abs(closest_angle) < (math.pi/4)):
                detection = self.get_latest_detection()
                
                if detection == 0:  # empty
                    self.get_logger().info('   No sign detected → State 1 (Forward)')
                    self.state = 1
                    self.movement_start = True
                elif detection == 2:  # right
                    self.get_logger().info('   RIGHT sign → State 2 (Turn Right 90°)')
                    self.state = 2
                    self.movement_start = True
                elif detection == 1:  # left
                    self.get_logger().info('   LEFT sign → State 3 (Turn Left 90°)')
                    self.state = 3
                    self.movement_start = True
                elif detection == 3 or detection == 4:  # do_not_enter or stop
                    self.get_logger().info('   STOP/DNE sign → State 4 (Turn Around)')
                    self.state = 4
                    self.movement_start = True
                elif detection == 5:  # goal
                    self.get_logger().info('   GOAL sign → State 6 (Goal Reached!)')
                    self.state = 6
            else:
                self.state = 1
                self.movement_start = True
        
        # ========== State 1: Move Forward ==========
        elif self.state == 1:
            if self.forward_distance < 0.55:
                self.get_logger().info('State 1: The way is blocked, stopping...')
                msg.linear.x = 0.0
                msg.linear.y = 0.0
                msg.linear.z = 0.0
                msg.angular.x = 0.0
                msg.angular.y = 0.0
                msg.angular.z = 0.0
                
                # Check for sign detection
                detection = self.get_latest_detection()

                if detection == -1:
                    self.get_logger().info('   No valid detection yet, waiting...')
                    if self.right_distance > self.left_distance:
                        self.get_logger().info(f'   → Right wider → State 7 (Small Right Turn)')
                        self.state = 7
                    else:
                        self.get_logger().info(f'   → Left wider → State 8 (Small Left Turn)')
                        self.state = 8
                
                if detection == 0:  # empty
                    self.get_logger().info(
                        f'   No sign. Left={self.left_distance:.2f}m, Right={self.right_distance:.2f}m'
                    )
                    
                    # Use small angle turns for obstacle avoidance
                    if self.right_distance > self.left_distance:
                        self.get_logger().info(f'   → Right wider → State 7 (Small Right Turn)')
                        self.state = 7
                    else:
                        self.get_logger().info(f'   → Left wider → State 8 (Small Left Turn)')
                        self.state = 8
                
                elif detection == 2:  # right
                    self.get_logger().info('   RIGHT sign → State 2 (Turn Right 90°)')
                    self.state = 2
                elif detection == 1:  # left
                    self.get_logger().info('   LEFT sign → State 3 (Turn Left 90°)')
                    self.state = 3
                elif detection == 3 or detection == 4:  # do_not_enter or stop
                    self.get_logger().info('   STOP/DNE sign → State 4 (Turn Around)')
                    self.state = 4
                elif detection == 5:  # goal
                    self.get_logger().info('   GOAL sign → State 6 (Goal Reached!)')
                    self.state = 6
                
                self.movement_start = True
            else:
                # Forward movement with wall following
                msg.linear.x = 0.1
                msg.linear.y = 0.0
                msg.linear.z = 0.0
                msg.angular.x = 0.0
                msg.angular.y = 0.0
                angular_velocity = 0.0
                
                # Left wall following
                if self.left_distance < 0.6 and abs(self.left_scan[0, 1] - self.left_scan[-1, 1]) < 0.4:
                    closest_left_angle = self.left_scan[np.argmin(self.left_scan[:, 1]), 0]
                    angular_velocity += 0.5 * (closest_left_angle - (math.pi / 2))
                
                # Right Wall Following
                elif self.right_distance < 0.6 and abs(self.right_scan[0, 1] - self.right_scan[-1, 1]) < 0.4:
                    closest_right_angle = self.right_scan[np.argmin(self.right_scan[:, 1]), 0]
                    angular_velocity -= 0.5 * (-(math.pi/2) - closest_right_angle)

                
                # Clamp angular velocity
                if angular_velocity > 0.2:
                    angular_velocity = 0.2
                if angular_velocity < -0.2:
                    angular_velocity = -0.2
                msg.angular.z = angular_velocity
                self.get_logger().info(f'Angular Velocity: {angular_velocity}')
        
        # ========== State 2: Turn Right 90° ==========
        elif self.state == 2:
            target_angle = -1 * math.pi / 2  # -90 degrees
            d_theta = target_angle - self.globalAng
            d_theta = math.atan2(math.sin(d_theta), math.cos(d_theta))
            angular_velocity = self.angular_pid.compute(d_theta, time.time())
            self.get_logger().info(
                f'State 2: Turning Right 90° (remaining: {math.degrees(d_theta):.1f}°)'
            )
            
            if abs(d_theta) > 0.05:
                msg.linear.x = 0.0
                msg.linear.y = 0.0
                msg.linear.z = 0.0
                msg.angular.x = 0.0
                msg.angular.y = 0.0
                if angular_velocity >= 0:
                    msg.angular.z = min(angular_velocity, self.max_angular_speed)
                else:
                    msg.angular.z = max(angular_velocity, -self.max_angular_speed)
            else:
                msg.linear.x = 0.0
                msg.linear.y = 0.0
                msg.linear.z = 0.0
                msg.angular.x = 0.0
                msg.angular.y = 0.0
                msg.angular.z = 0.0
                self.get_logger().info('   Turn complete → State 5 (Pause)')
                self.state = 5
        
        # ========== State 3: Turn Left 90° ==========
        elif self.state == 3:
            target_angle = math.pi / 2  # 90 degrees
            d_theta = target_angle - self.globalAng
            d_theta = math.atan2(math.sin(d_theta), math.cos(d_theta))
            angular_velocity = self.angular_pid.compute(d_theta, time.time())
            self.get_logger().info(
                f'State 3: Turning Left 90° (remaining: {math.degrees(d_theta):.1f}°)'
            )
            
            if abs(d_theta) > 0.1:
                msg.linear.x = 0.0
                msg.linear.y = 0.0
                msg.linear.z = 0.0
                msg.angular.x = 0.0
                msg.angular.y = 0.0
                if angular_velocity >= 0:
                    msg.angular.z = min(angular_velocity, self.max_angular_speed)
                else:
                    msg.angular.z = max(angular_velocity, -self.max_angular_speed)
            else:
                msg.linear.x = 0.0
                msg.linear.y = 0.0
                msg.linear.z = 0.0
                msg.angular.x = 0.0
                msg.angular.y = 0.0
                msg.angular.z = 0.0
                self.get_logger().info('   Turn complete → State 5 (Pause)')
                self.state = 5
        
        # ========== State 4: Turn Around 180° ==========
        elif self.state == 4:
            target_angle = math.pi  # 180 degrees
            d_theta = target_angle - self.globalAng
            d_theta = math.atan2(math.sin(d_theta), math.cos(d_theta))
            angular_velocity = self.angular_pid.compute(d_theta, time.time())
            self.get_logger().info(
                f'State 4: Turning Around 180° (remaining: {math.degrees(d_theta):.1f}°)'
            )
            
            if abs(d_theta) > 0.1:
                msg.linear.x = 0.0
                msg.linear.y = 0.0
                msg.linear.z = 0.0
                msg.angular.x = 0.0
                msg.angular.y = 0.0
                if angular_velocity >= 0:
                    msg.angular.z = min(angular_velocity, self.max_angular_speed)
                else:
                    msg.angular.z = max(angular_velocity, -self.max_angular_speed)
            else:
                msg.linear.x = 0.0
                msg.linear.y = 0.0
                msg.linear.z = 0.0
                msg.angular.x = 0.0
                msg.angular.y = 0.0
                msg.angular.z = 0.0
                self.get_logger().info('   Turn around complete → State 5 (Pause)')
                self.state = 5
        
        # ========== State 5: Pause ==========
        elif self.state == 5:
            self.get_logger().info('State 5: Pausing...')
            msg.linear.x = 0.0
            msg.linear.y = 0.0
            msg.linear.z = 0.0
            msg.angular.x = 0.0
            msg.angular.y = 0.0
            msg.angular.z = 0.0
            time.sleep(0.5)
            self.movement_start = True
            self.get_logger().info('   → State 1 (Forward)')
            self.state = 1
        
        # ========== State 6: Goal Reached ==========
        elif self.state == 6:
            msg.linear.x = 0.0
            msg.linear.y = 0.0
            msg.linear.z = 0.0
            msg.angular.x = 0.0
            msg.angular.y = 0.0
            msg.angular.z = 0.0
            self.get_logger().info('🎯 State 6: GOAL REACHED! Mission complete!')
            time.sleep(0.5)
            self.state = 5
        
        # ========== State 7: Small Turn Right ==========
        elif self.state == 7:
            target_angle = -1 * self.small_turn_angle  # Negative for right turn
            d_theta = target_angle - self.globalAng
            d_theta = math.atan2(math.sin(d_theta), math.cos(d_theta))
            angular_velocity = self.angular_pid.compute(d_theta, time.time())
            self.get_logger().info(
                f'State 7: Small Right Turn {math.degrees(self.small_turn_angle):.0f}° '
                f'(remaining: {math.degrees(d_theta):.1f}°)'
            )
            
            if abs(d_theta) > 0.05:
                msg.linear.x = 0.0
                msg.linear.y = 0.0
                msg.linear.z = 0.0
                msg.angular.x = 0.0
                msg.angular.y = 0.0
                if angular_velocity >= 0:
                    msg.angular.z = min(angular_velocity, self.max_angular_speed)
                else:
                    msg.angular.z = max(angular_velocity, -self.max_angular_speed)
            else:
                msg.linear.x = 0.0
                msg.linear.y = 0.0
                msg.linear.z = 0.0
                msg.angular.x = 0.0
                msg.angular.y = 0.0
                msg.angular.z = 0.0
                self.get_logger().info('   Small turn complete → State 5 (Pause)')
                self.state = 5
        
        # ========== State 8: Small Turn Left ==========
        elif self.state == 8:
            target_angle = self.small_turn_angle  # Positive for left turn
            d_theta = target_angle - self.globalAng
            d_theta = math.atan2(math.sin(d_theta), math.cos(d_theta))
            angular_velocity = self.angular_pid.compute(d_theta, time.time())
            self.get_logger().info(
                f'State 8: Small Left Turn {math.degrees(self.small_turn_angle):.0f}° '
                f'(remaining: {math.degrees(d_theta):.1f}°)'
            )
            
            if abs(d_theta) > 0.05:
                msg.linear.x = 0.0
                msg.linear.y = 0.0
                msg.linear.z = 0.0
                msg.angular.x = 0.0
                msg.angular.y = 0.0
                if angular_velocity >= 0:
                    msg.angular.z = min(angular_velocity, self.max_angular_speed)
                else:
                    msg.angular.z = max(angular_velocity, -self.max_angular_speed)
            else:
                msg.linear.x = 0.0
                msg.linear.y = 0.0
                msg.linear.z = 0.0
                msg.angular.x = 0.0
                msg.angular.y = 0.0
                msg.angular.z = 0.0
                self.get_logger().info('   Small turn complete → State 5 (Pause)')
                self.state = 5
        
        # Publish velocity command
        self.vel_publisher.publish(msg)
    
    def update_Odometry(self, Odom):
        """Update odometry information"""
        position = Odom.pose.pose.position
        
        # Orientation uses quaternion parametrization
        # To get angular position along z-axis:
        q = Odom.pose.pose.orientation
        orientation = np.arctan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
        
        if self.movement_start:
            # Store initial data to subtract from all other values
            # Start at position (0,0) and orientation 0
            self.movement_start = False
            self.Init_ang = orientation
            self.globalAng = self.Init_ang
            Mrot = np.matrix([
                [np.cos(self.Init_ang), np.sin(self.Init_ang)],
                [-np.sin(self.Init_ang), np.cos(self.Init_ang)]
            ])
            self.Init_pos.x = Mrot.item((0,0))*position.x + Mrot.item((0,1))*position.y
            self.Init_pos.y = Mrot.item((1,0))*position.x + Mrot.item((1,1))*position.y
            self.Init_pos.z = position.z
        
        Mrot = np.matrix([
            [np.cos(self.Init_ang), np.sin(self.Init_ang)],
            [-np.sin(self.Init_ang), np.cos(self.Init_ang)]
        ])
        
        # Subtract initial values
        self.globalPos.x = Mrot.item((0,0))*position.x + Mrot.item((0,1))*position.y - self.Init_pos.x
        self.globalPos.y = Mrot.item((1,0))*position.x + Mrot.item((1,1))*position.y - self.Init_pos.y
        globalAng = orientation - self.Init_ang
        self.globalAng = math.atan2(math.sin(globalAng), math.cos(globalAng))


def main(args=None):
    rclpy.init(args=args)
    move_robot = MoveRobot()
    
    try:
        rclpy.spin(move_robot)
    except KeyboardInterrupt:
        pass
    finally:
        move_robot.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
