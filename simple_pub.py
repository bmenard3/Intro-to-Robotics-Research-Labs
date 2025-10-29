import rclpy
from rclpy.node import Node
from std_msgs import Bool

class SimplePub(Node):
    def __init__(self):
        super().__init__('simple_publisher')
        self.publisher = self.create_publisher(Bool, 'amazing_bool', 10)


def main(args=None):
    rclpy.init(args=args)
    simple_publisher = SimplePub()
    rclpy.spin(minimal_publisher)

    minimal_publisher.destroy_node()
    rclpy.shutdown()


