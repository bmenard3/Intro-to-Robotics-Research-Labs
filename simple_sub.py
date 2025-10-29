import rclpy
from rclpy.node import Node
from std_msgs import UInt64

class SimpleSub(Node):
    def __init__(self):
        super().__init__('simple_subscriber')
        self.subscription = self.create_subscription(
            UInt64,
            self.magic_fun,
            10
        )

def main(args=None):
    rclpy.init(args=args)
    simple_subscriber = SimpleSub()
    rclpy.spin(minimal_subscriber)

    minimal_subscriber.destroy_node()
    rclpy.shutdown()


