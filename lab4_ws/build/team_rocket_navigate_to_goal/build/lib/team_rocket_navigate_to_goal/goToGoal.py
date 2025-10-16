import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Point
from std_msgs.msg import Float32
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

class GoToGoal(Node):
    def __init__(self):
        super().__init__('go_to_goal')
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth = 1
        )
        self.range_subscriber = self.create_subscriber = self.create_subscription(
            Float32,
            '/obstacle_detection',
            self.range_callback,
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
            'cmd_vel',
            10
        )
        timer_period = 0.1
        self.timer = self.create_timer(timer_period, self.navigation)
        self.Init = True
        self.Init_pos = Point()
        self.Init_pos.x = 0.0
        self.Init_pos.y = 0.0
        self.Init_ang = 0.0
        self.globalPos = Point()
        self.globalAng = 0.0
        self.state = 0
        self.wait_start = 0

        #self.waypoints = np.array([[1.5, 0], [1.5, 1.4], [0, 1.4]])
        self.waypoints = np.array([[1.0, 0], [0.0, 1.4]])
        self.current_goal = 0
        self.goal_pos = Point()
        self.goal_pos.x = 0.0
        self.goal_pos.y = 0.0

        self.angular_pid = PIDController(
            kp=4.0,
            ki=1.2,
            kd=1.0,
            integral_limit=0.5
        )
        self.max_angular_speed = 1.0

        self.get_logger().info(f'Initialized with state {self.state}')
        
    def odom_callback(self, msg):
        self.update_Odometry(msg)

    def range_callback(self, msg):
        a = msg
    
    def navigation(self):
        msg = Twist()
        if self.state == 0: # Go To Goal
            self.get_logger().info(f'Going to Goal')
            self.goal_pos.x = self.waypoints[self.current_goal, 0]
            self.goal_pos.y = self.waypoints[self.current_goal, 1]
            dx = self.goal_pos.x - self.globalPos.x
            dy = self.goal_pos.y - self.globalPos.y
            d = math.sqrt(dx**2 + dy**2)
            target_angle = math.atan2(dy, dx)
            d_theta = target_angle - self.globalAng
            angular_velocity = self.angular_pid.compute(d_theta, time.time())
            if d > 0.05:
                msg.linear.x = 1.0
                msg.linear.y = 0.0
                msg.linear.z = 0.0
                msg.angular.x = 0.0
                msg.angular.y = 0.0
                msg.angular.z = angular_velocity
            else:
                msg.linear.x = 0.0
                msg.linear.y = 0.0
                msg.linear.z = 0.0
                msg.angular.x = 0.0
                msg.angular.y = 0.0
                msg.angular.z = 0.0
                self.state = 2
                self.wait_start = time.time()

        elif self.state == 1: # Avoid Obstacles
            msg.linear.x = 0.0
            msg.linear.y = 0.0
            msg.linear.z = 0.0
            msg.angular.x = 0.0
            msg.angular.y = 0.0
            msg.angular.z = 0.0

        elif self.state == 2: # Wait At Goal
            self.get_logger().info(f'Waiting at Goal')
            msg.linear.x = 0.0
            msg.linear.y = 0.0
            msg.linear.z = 0.0
            msg.angular.x = 0.0
            msg.angular.y = 0.0
            msg.angular.z = 0.0
            if (time.time() - self.wait_start) > 10:
                self.state = 0
                self.current_goal += 1
        
        self.vel_publisher.publish(msg)

    def update_Odometry(self, Odom):
        position = Odom.pose.pose.position
        
        #Orientation uses the quaternion aprametrization.
        #To get the angular position along the z-axis, the following equation is required.
        q = Odom.pose.pose.orientation
        orientation = np.arctan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))

        if self.Init:
            #The initial data is stored to by subtracted to all the other values as we want to start at position (0,0) and orientation 0
            self.Init = False
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
        self.globalAng = orientation - self.Init_ang
        self.get_logger().info(f'Current Position: x = {self.globalPos.x}, y = {self.globalPos.y}, theta = {self.globalAng}')

def main(args=None):
    rclpy.init(args=args)
    go_to_goal = GoToGoal()
    rclpy.spin(go_to_goal)

    go_to_goal.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
    
    
