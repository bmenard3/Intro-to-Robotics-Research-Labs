from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='team_rocket_final',
            executable='moveRobot',
            name='move_robot'
        ),
        Node(
            package='team_rocket_final',
            executable='getObjectRange',
            name='get_object_range'
        )
    ])

