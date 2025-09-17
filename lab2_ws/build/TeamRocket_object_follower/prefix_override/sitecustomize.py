import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/augustmenard/Intro-to-Robotics-Research-Labs/lab2_ws/install/TeamRocket_object_follower'
