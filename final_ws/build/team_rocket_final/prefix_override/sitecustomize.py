import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/j/Desktop/Leo/Intro-to-Robotics-Research-Labs/final_ws/install/team_rocket_final'
