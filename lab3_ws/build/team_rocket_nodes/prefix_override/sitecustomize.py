import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/michaelangelo/teamRocket/Intro-to-Robotics-Research-Labs/lab3_ws/install/team_rocket_nodes'
