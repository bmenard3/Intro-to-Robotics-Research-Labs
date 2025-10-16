from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='team_rocket_navigate_to_goal',
            executable='getObjectRange',
            name='get_object_range'
        ),
        Node(
            package='team_rocket_navigate_to_goal',
            executable='goToGoal',
            name='go_to_goal' 
        )
    ])
    