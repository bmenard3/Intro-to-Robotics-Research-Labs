#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped, PointStamped
from rclpy.duration import Duration
import time
import math


class DynamicWaypointNavigator(Node):
    def __init__(self):
        super().__init__('dynamic_waypoint_navigator')
        
        self._action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        
        self.clicked_point_sub = self.create_subscription(
            PointStamped,
            '/clicked_point',
            self.clicked_point_callback,
            10
        )
        
        self.waypoints = []
        self.current_waypoint_index = 0
        self.goal_handle = None
        self.is_navigating = False
        
        self.get_logger().info('Dynamic Waypoint Navigator initialized!')
        self.get_logger().info('Click points in RViz using "Publish Point" tool to add waypoints')
        self.get_logger().info('Waypoints will be added to the queue and navigated automatically')
    
    def clicked_point_callback(self, msg):
        x = msg.point.x
        y = msg.point.y
        yaw = 0.0
        
        self.waypoints.append((x, y, yaw))
        
        self.get_logger().info(f'New waypoint added: x={x:.2f}, y={y:.2f}')
        self.get_logger().info(f'Total waypoints in queue: {len(self.waypoints)}')
        
        if not self.is_navigating:
            self.get_logger().info('Starting navigation...')
            self.send_goal()
    
    def create_pose_stamped(self, x, y, yaw):
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = 0.0
        
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = math.sin(yaw / 2.0)
        pose.pose.orientation.w = math.cos(yaw / 2.0)
        
        return pose
    
    def send_goal(self):
        if self.current_waypoint_index >= len(self.waypoints):
            self.get_logger().info('All waypoints completed! 🎉')
            self.get_logger().info('Add more waypoints by clicking in RViz')
            self.is_navigating = False
            return False
        
        self.is_navigating = True
        
        self.get_logger().info('Waiting for action server...')
        self._action_client.wait_for_server()
        
        x, y, yaw = self.waypoints[self.current_waypoint_index]
        
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = self.create_pose_stamped(x, y, yaw)
        
        self.get_logger().info(f'Navigating to waypoint {self.current_waypoint_index + 1}/{len(self.waypoints)}: '
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
            self.is_navigating = False
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
                self.get_logger().info('All current waypoints completed!')
        else:
            self.get_logger().error(f'✗ Navigation to waypoint {self.current_waypoint_index + 1} failed with status: {status}')
            self.get_logger().error('Stopping navigation.')
            self.is_navigating = False


def main(args=None):
    rclpy.init(args=args)
    
    navigator = DynamicWaypointNavigator()
    
    time.sleep(1.0)
    
    try:
        rclpy.spin(navigator)
    except KeyboardInterrupt:
        navigator.get_logger().info('Navigation interrupted by user')
    
    navigator.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()