import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
import numpy as np

class GetObjectRange(Node):
    def __init__(self):
        super().__init__('get_object_range')
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth = 1
        )  
        self.scan_subscriber = self.create_subscriber = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            qos_profile
        )
        self.publisher_ = self.create_publisher(
            Float32,
            '/obstacle_detection',
            10
        )
        timer_period = 0.1
        self.timer = self.create_timer(timer_period, self.timer_callback)


    def scan_callback(self, msg):
        self.angle_min = msg.angle_min
        self.angle_max = msg.angle_max
        self.angle_increment = msg.angle_increment
        self.ranges = msg.ranges
        self.angles = list(np.arange(self.angle_min, self.angle_max, self.angle_increment))
	min_index = self.ranges.index(min(self.ranges))
        self.get_logger().info(f'min range = {self.ranges[min_index]}, min_angle = {self.angles[min_index]')
    
    def timer_callback(self):
        msg = Float32()
        msg.data = 0.0
        


def main(args=None):
    rclpy.init(args=args)
    get_object_range = GetObjectRange()
    rclpy.spin(get_object_range)

    get_object_range.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
