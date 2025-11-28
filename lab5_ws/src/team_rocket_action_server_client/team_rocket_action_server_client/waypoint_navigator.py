#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped
from rclpy.duration import Duration
import time


class WaypointNavigator(Node):
    def __init__(self):
        super().__init__('waypoint_navigator')
        
        self._action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        
        
        self.waypoints = [
            (0.4, 1.0, 0.0),
            (1.3, 1.0, 0.0),
            (0.1, -1.0, 0.0),
        ]
        
        self.current_waypoint_index = 0
        self.goal_handle = None
        
        self.get_logger().info('Waypoint Navigator initialized!')
        self.get_logger().info(f'Total waypoints to visit: {len(self.waypoints)}')
    
    def quaternion_from_euler(self, roll, pitch, yaw):
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        q = [0.0, 0.0, 0.0, 0.0]
        q[0] = cy * cp * cr + sy * sp * sr
        q[1] = cy * cp * sr - sy * sp * cr
        q[2] = sy * cp * sr + cy * sp * cr
        q[3] = sy * cp * cr - cy * sp * sr

        return q
    
    def create_pose_stamped(self, x, y, yaw):
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = 0.0
        
        import math
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = math.sin(yaw / 2.0)
        pose.pose.orientation.w = math.cos(yaw / 2.0)
        
        return pose
    
    def send_goal(self):
        if self.current_waypoint_index >= len(self.waypoints):
            self.get_logger().info('All waypoints completed! 🎉')
            return False
        
        self.get_logger().info('Waiting for action server...')
        self._action_client.wait_for_server()
        
        x, y, yaw = self.waypoints[self.current_waypoint_index]
        
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = self.create_pose_stamped(x, y, yaw)
        
        self.get_logger().info(f'Sending goal {self.current_waypoint_index + 1}/{len(self.waypoints)}: '
                               f'x={x:.2f}, y={y:.2f}, yaw={yaw:.2f} rad')
        
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        
        self._send_goal_future.add_done_callback(self.goal_response_callback)
        
        return True
    
    def goal_response_callback(self, future):
        self.goal_handle = future.result()
        
        if not self.goal_handle.accepted:
            self.get_logger().error('Goal rejected!')
            return
        
        self.get_logger().info('Goal accepted! Navigating...')
        
        self._get_result_future = self.goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)
    
    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        pass
    
    def get_result_callback(self, future):
        result = future.result().result
        status = future.result().status
        
        if status == 4:
            self.get_logger().info(f'✓ Waypoint {self.current_waypoint_index + 1} reached successfully!')
            
            self.current_waypoint_index += 1
            
            time.sleep(1.0)
            
            if not self.send_goal():
                self.get_logger().info('Mission complete! All waypoints visited.')
        else:
            self.get_logger().error(f'✗ Navigation to waypoint {self.current_waypoint_index + 1} failed with status: {status}')
            self.get_logger().error('Stopping navigation. You can modify this to retry or continue.')


def main(args=None):
    rclpy.init(args=args)
    
    navigator = WaypointNavigator()
    
    time.sleep(2.0)
    
    if navigator.send_goal():
        try:
            rclpy.spin(navigator)
        except KeyboardInterrupt:
            navigator.get_logger().info('Navigation interrupted by user')
    
    navigator.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()