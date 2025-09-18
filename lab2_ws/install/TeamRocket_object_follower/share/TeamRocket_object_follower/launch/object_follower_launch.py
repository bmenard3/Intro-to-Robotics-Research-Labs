from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='TeamRocket_object_follower',
            executable='find_object',
            name='find_object'
        ),
        Node(
            package='TeamRocket_object_follower',
            executable='rotate_robot',
            name='rotate_robot'
        )
    ])