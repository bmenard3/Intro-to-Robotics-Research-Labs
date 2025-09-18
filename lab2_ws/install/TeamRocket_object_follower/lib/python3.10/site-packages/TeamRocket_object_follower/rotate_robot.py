import rclpy
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from geometry_msgs.msg import Point, Twist

class RotateRobot(Node):
    def __init__(self):
        super().__init__('rotate_robot')
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth = 1
        )
        self.subscriber_ = self.create_subscriber = self.create_subscription(
            Point,
            'object_coordinates',
            self.coordinate_callback,
            qos_profile)
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        timer_period = 0.1
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.dx = 0
        self.dy = 0

        
    def coordinate_callback(self, msg):
        self.dx = msg.x
        self.dy = msg.y


    def timer_callback(self):
        msg = Twist()
        msg.linear.x = 0.0
        msg.linear.y = 0.0
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = self.dx / 100
        self.publisher_.publish(msg)
    

def main(args=None):
    rclpy.init(args=args)
    rotate_robot = RotateRobot()
    rclpy.spin(rotate_robot)
    rotate_robot.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()    

