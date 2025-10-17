import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/augustmenard/Intro-to-Robotics-Research-Labs/lab4_ws/install/team_rocket_navigate_to_goal'
