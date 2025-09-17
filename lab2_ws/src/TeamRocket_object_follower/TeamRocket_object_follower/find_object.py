import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from geometry_msgs.msg import Point
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np
from PIL import Image

class FindObject(Node):
    def __init__(self):
        super().__init__('find_object')
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth = 1
        )
        self._img_subscriber = self._img_subscriber = self.create_subscription(
            CompressedImage,
            '/image_raw/compressed',
            self._image_callback,
            qos_profile)
        self.publisher_ = self.create_publisher(Point, 'object_coordinates', 10)
        timer_period = 0.1
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.colormask_lower_limit = np.array([100,75,75])
        self.colormask_upper_limit = np.array([120,200,200])
        self.dx = 0
        self.dy = 0
    

    def _image_callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        y, x = hsv.shape[:2]
        xc = x // 2
        yc = y // 2
        mask = cv2.inRange(hsv, self.colormask_lower_limit, self.colormask_upper_limit)
        mask_img = Image.fromarray(mask)
        bbox = mask_img.getbbox()
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            xt = int((x1 + x2) / 2)
            yt = int((y1 + y2) / 2)
        else:
            xt = xc
            yt = yc
        self.dx = xc - xt
        self.dy = yc - yt
        
        
    def timer_callback(self):
        msg = Point()
        msg.x = float(self.dx)
        msg.y = float(self.dy)
        msg.z = 0.0
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing dx = {self.dx}')
    

def main(args=None):
    rclpy.init(args=args)
    find_object = FindObject()
    rclpy.spin(find_object)

    find_object.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

