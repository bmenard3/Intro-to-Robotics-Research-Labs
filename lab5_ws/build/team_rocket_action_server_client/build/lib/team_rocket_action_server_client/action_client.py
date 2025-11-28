import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateThroughPoses
import geometry_msgs

class Nav2ActionClient(Node):
    def __init__(self):
        super().__init__('action_client')
        self.action_client = ActionClient(self, NavigateThroughPoses, 'navigate_through_poses')
    
    def send_goal(self):
        self.get_logger().info('Running send_goal()')
        goal_msg = NavigateThroughPoses.Goal()
        goal_msg.poses = []
        pose1 = geometry_msgs.msg.PoseStamped()
        pose1.header.frame_id = 'sim_map'
        pose1.pose.position.x = 0.0
        pose1.pose.position.y = 2.0
        pose1.pose.orientation.w = 0.0
        pose2 = geometry_msgs.msg.PoseStamped()
        pose2.header.frame_id = 'sim_map'
        pose2.pose.position.x = 0.0
        pose2.pose.position.y = 0.0
        pose2.pose.orientation.w = 0.0
        pose3 = geometry_msgs.msg.PoseStamped()
        pose3.header.frame_id = 'sim_map'
        pose3.pose.position.x = 0.0
        pose3.pose.position.y = 2.0
        pose3.pose.orientation.w = 0.0
        goal_msg.poses.append(pose1)
        goal_msg.poses.append(pose2)
        goal_msg.poses.append(pose3)
        self.get_logger().info(f'Sending {len(goal_msg.poses)} Points')
        
        self.action_client.wait_for_server()
        self.get_logger().info('Waited for server')
        future = self.action_client.send_goal_async(
            goal_msg, feedback_callback=self.feedback_callback)
        self.get_logger().info('Got future object')
        return future
        
    def feedback_callback(self, feedback_msg):
        self.get_logger().info(f'Received feedback: {feedback_msg.feedback}')

def main(args=None):
    rclpy.init(args=args)
    client = Nav2ActionClient()
    client.send_goal()
    rclpy.spin(client)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
