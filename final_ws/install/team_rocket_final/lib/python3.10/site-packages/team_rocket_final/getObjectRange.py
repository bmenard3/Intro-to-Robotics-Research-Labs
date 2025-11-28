import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray, MultiArrayLayout, MultiArrayDimension
import numpy as np

class GetObjectRange(Node):
    def __init__(self):
        super().__init__('get_object_range')
        self.angles = None
        self.ranges = None
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
        self.range_publisher = self.create_publisher(
            Float32MultiArray,
            '/ranges',
            10
        )
        self.angle_publisher = self.create_publisher(
            Float32MultiArray,
            '/angles',
            10
        )
        self.scan_publisher = self.create_publisher(
            Float32MultiArray,
            '/processed_scans',
            10
        )
        timer_period = 0.1
        self.timer = self.create_timer(timer_period, self.timer_callback)


    def scan_callback(self, msg):
        self.angle_min = msg.angle_min
        self.angle_max = msg.angle_max
        self.angle_increment = msg.angle_increment
        self.ranges = np.array(msg.ranges, dtype=np.float32)
        self.angles = np.arange(self.angle_min, self.angle_max, self.angle_increment)
        self.angles = (self.angles + np.pi) % (2*np.pi) - np.pi
        
    
    def timer_callback(self):
        if self.angles is not None and self.ranges is not None:
            scan_msg = Float32MultiArray()
            scan_msg_data = np.hstack([np.atleast_2d(self.angles).T, np.atleast_2d(self.ranges).T])
            scan_msg_data = scan_msg_data[scan_msg_data[:,0].argsort()]
            N, M = scan_msg_data.shape
            scan_msg.layout = MultiArrayLayout(
                dim=[
                    MultiArrayDimension(label='rows', size=N, stride=N * M),
                    MultiArrayDimension(label='cols', size=M, stride=M)
                ],
                data_offset=0
            )
            scan_msg.data = scan_msg_data.astype(np.float32).flatten().tolist()
            angle_msg = Float32MultiArray()
            angle_msg.data = self.angles.astype(np.float32).flatten().tolist()
            range_msg = Float32MultiArray()
            range_msg.data = self.ranges.tolist()
            self.angle_publisher.publish(angle_msg)
            self.range_publisher.publish(range_msg)
            self.scan_publisher.publish(scan_msg)
        


def main(args=None):
    rclpy.init(args=args)
    get_object_range = GetObjectRange()
    rclpy.spin(get_object_range)

    get_object_range.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()