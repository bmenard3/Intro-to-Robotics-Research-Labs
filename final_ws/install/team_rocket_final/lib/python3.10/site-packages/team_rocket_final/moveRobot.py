import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Point
from std_msgs.msg import Float32MultiArray
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
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth = 1
        )
        self.range_subscriber = self.create_subscriber = self.create_subscription(
            Float32MultiArray,
            '/ranges',
            self.range_callback,
            qos_profile
        )
        self.angle_subscriber = self.create_subscriber = self.create_subscription(
            Float32MultiArray,
            '/angles',
            self.angle_callback,
            qos_profile
        )
        self.scan_subscriber = self.create_subscriber = self.create_subscription(
            Float32MultiArray,
            '/processed_scans',
            self.scan_callback,
            qos_profile
        )
        self.odom_subscriber = self.create_subscriber = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            qos_profile
        )
        self.vel_publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )
        timer_period = 0.1
        self.timer = self.create_timer(timer_period, self.navigation)
        self.angular_pid = PIDController(
            kp=0.5,
            ki=0.0,
            kd=0.0,
            integral_limit=3.0
        )
        self.max_angular_speed = 1.0
        self.movement_start = False
        self.Init_pos = Point()
        self.Init_pos.x = 0.0
        self.Init_pos.y = 0.0
        self.Init_ang = 0.0
        self.globalPos = Point()
        self.globalAng = 0.0
        self.state = 1
        self.ranges = [0.5]
        self.angles = [math.pi]
        self.forward_distance = 1
        self.right_distance = 2
        self.left_distance = 2
        self.movement_goal = Point()
        self.movement_goal.x = 1.0
        self.movement_goal.y = 0.0

    def odom_callback(self, msg):
        self.update_Odometry(msg)

    def range_callback(self, msg):
        self.ranges = msg.data
    
    def angle_callback(self, msg):
        angles = np.array(msg.data)
        self.angles = list((angles + np.pi) % (2 * np.pi) - np.pi)
    
    def scan_callback(self, msg):
        N = msg.layout.dim[0].size
        M = msg.layout.dim[1].size
        self.latest_scan = np.array(msg.data, dtype=np.float32).reshape(N, M)
        for i in range(self.latest_scan.shape[0]):
            if np.isnan(self.latest_scan[i,1]):
                self.latest_scan[i,1] = 5.0
        forward_scan_ind0 = np.argmin(np.abs(self.latest_scan[:, 0] + (math.pi / 12)))
        forward_scan_ind1 = np.argmin(np.abs(self.latest_scan[:, 0] - (math.pi / 12)))

        self.forward_scan = self.latest_scan[forward_scan_ind0:forward_scan_ind1, :]
        #self.get_logger().info(f'ind0: {forward_scan_ind0}, ind1: {forward_scan_ind1}')
        self.forward_distance = np.min(self.forward_scan[:,1])
        #self.get_logger().info(f'Forward Distance: {self.forward_distance}')
        right_scan_ind0 = np.argmin(np.abs(self.latest_scan[:, 0] + ((math.pi / 2) + (math.pi / 6))))
        right_scan_ind1 = np.argmin(np.abs(self.latest_scan[:, 0] + ((math.pi / 2) - (math.pi / 6))))
        #self.get_logger().info(f'ind0: {right_scan_ind0}, ind1: {right_scan_ind1}')
        self.right_scan = self.latest_scan[right_scan_ind0:right_scan_ind1, :]
        #self.get_logger().info(f'{self.latest_scan}')
        self.right_distance = np.min(self.right_scan[:,1])
        left_scan_ind0 = np.argmin(np.abs(self.latest_scan[:, 0] - ((math.pi / 2) - (math.pi / 6))))
        left_scan_ind1 = np.argmin(np.abs(self.latest_scan[:, 0] - ((math.pi / 2) + (math.pi / 6))))
        self.left_scan = self.latest_scan[left_scan_ind0:left_scan_ind1, :]
        self.left_distance = np.min(self.left_scan[:,1])
        #self.get_logger().info(f'{self.left_scan}')
        #self.get_logger().info(f'right_distance: {self.right_distance}')


    def navigation(self):
        msg = Twist()
        closest_scan = self.ranges.index(min(self.ranges))
        closest_range = self.ranges[closest_scan]
        closest_angle = self.angles[closest_scan]
        if self.state == 0: # Detect
            self.get_logger().info('Initializing')
            if (closest_range < 0.5) and (abs(closest_angle) < (math.pi/4)):
                detection = 0
                if not detection:
                    self.state = 3
                    self.movement_start = True
                elif detection == 'right':
                    self.state = 2
                    self.movement_start = True
                elif detection == 'left':
                    self.state = 3
                    self.movement_start = True
                elif detection == 'stop' or detection == 'turn around':
                    self.state = 4
                    self.movement_start = True
                elif detection == 'goal':
                    self.state = 6
            else:
                self.state = 1
                self.movement_start = True

        elif self.state == 1: # Move Forward
            if self.forward_distance < 0.5:
                self.get_logger().info('The way is blocked')
                msg.linear.x = 0.0
                msg.linear.y = 0.0
                msg.linear.z = 0.0
                msg.angular.x = 0.0
                msg.angular.y = 0.0
                msg.angular.z = 0.0
                self.movement_start = True
                self.get_logger().info(f'left dist: {self.left_distance}, right dist: {self.right_distance}')
                
                if self.right_distance > self.left_distance:
                    self.state = 2
                    self.movement_start = True
                else:
                    self.state = 3
                    self.movement_start = True
            else:
                msg.linear.x = 0.1
                msg.linear.y = 0.0
                msg.linear.z = 0.0
                msg.angular.x = 0.0
                msg.angular.y = 0.0
                angular_velocity = 0.0
                if self.left_distance < 0.6:
                    closest_left_angle = self.left_scan[np.argmin(self.left_scan[:,1]), 0]
                    self.get_logger().info(f'closest left angle: {closest_left_angle}')
                    angular_velocity += 0.1*(closest_left_angle - (math.pi/2))
                if self.right_distance < 0.6:
                    closest_right_angle = self.right_scan[np.argmin(self.right_scan[:,1]), 0]
                    self.get_logger().info(f'closest right angle: {-(math.pi/2) - closest_right_angle}')
                    angular_velocity += 0.1*(-(math.pi/2) - closest_right_angle)
                    #self.get_logger().info(f'{self.right_scan[0:3,:]}')
                if angular_velocity > 0.2:
                    angular_velocity = 0.2
                if angular_velocity < -0.2:
                    angular_velocity = -0.2
                msg.angular.z = angular_velocity
                #self.get_logger().info(f'Angular Velocity = {msg.angular.z}')
                #self.get_logger().info(f'Right Distance: {self.right_distance}, Left Distance: {self.left_distance}')
        
        elif self.state == 2: # Turn Right
            target_angle = -1 * math.pi / 2
            d_theta = target_angle - self.globalAng
            d_theta = math.atan2(math.sin(d_theta), math.cos(d_theta))
            angular_velocity = self.angular_pid.compute(d_theta, time.time())
            self.get_logger().info(f'Turning Right')
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
                self.state = 5
        
        elif self.state == 3: # Turn Left
            self.get_logger().info('Turning Left')
            target_angle = math.pi / 2
            d_theta = target_angle - self.globalAng
            d_theta = math.atan2(math.sin(d_theta), math.cos(d_theta))
            angular_velocity = self.angular_pid.compute(d_theta, time.time())
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
                self.state = 5
        
        elif self.state == 4: # Turn Around
            self.get_logger().info('Turning Left')
            target_angle = math.pi / 2
            d_theta = target_angle - self.globalAng
            d_theta = math.atan2(math.sin(d_theta), math.cos(d_theta))
            angular_velocity = self.angular_pid.compute(d_theta, time.time())
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
                self.state = 0
        
        elif self.state == 5: # Pause
            self.get_logger().info('Pausing')
            time.sleep(0.5)
            self.movement_start = True
            self.state = 1
        
        elif self.state == 6: # Goal Reached
            self.get_logger().info('Goal Reached!')
        
        self.vel_publisher.publish(msg)
        

            
    def update_Odometry(self, Odom):
        position = Odom.pose.pose.position
        
        #Orientation uses the quaternion aprametrization.
        #To get the angular position along the z-axis, the following equation is required.
        q = Odom.pose.pose.orientation
        orientation = np.arctan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))

        if self.movement_start:
            #The initial data is stored to by subtracted to all the other values as we want to start at position (0,0) and orientation 0
            self.movement_start = False
            self.Init_ang = orientation
            self.globalAng = self.Init_ang
            Mrot = np.matrix([[np.cos(self.Init_ang), np.sin(self.Init_ang)],[-np.sin(self.Init_ang), np.cos(self.Init_ang)]])        
            self.Init_pos.x = Mrot.item((0,0))*position.x + Mrot.item((0,1))*position.y
            self.Init_pos.y = Mrot.item((1,0))*position.x + Mrot.item((1,1))*position.y
            self.Init_pos.z = position.z
        Mrot = np.matrix([[np.cos(self.Init_ang), np.sin(self.Init_ang)],[-np.sin(self.Init_ang), np.cos(self.Init_ang)]])        

        #We subtract the initial values
        self.globalPos.x = Mrot.item((0,0))*position.x + Mrot.item((0,1))*position.y - self.Init_pos.x
        self.globalPos.y = Mrot.item((1,0))*position.x + Mrot.item((1,1))*position.y - self.Init_pos.y
        globalAng = orientation - self.Init_ang
        self.globalAng = math.atan2(math.sin(globalAng), math.cos(globalAng))
        #self.get_logger().info(f'Current Position: x = {self.globalPos.x}, y = {self.globalPos.y}, theta = {self.globalAng}')


def main(args=None):
    rclpy.init(args=args)
    move_robot = MoveRobot()
    rclpy.spin(move_robot)

    move_robot.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
