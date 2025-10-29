import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/augustmenard/Intro-to-Robotics-Research-Labs/lab5_ws/install/turtlebot3_example'
