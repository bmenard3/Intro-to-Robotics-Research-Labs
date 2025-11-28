from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'team_rocket_final'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', 'team_rocket_final', 'launch'), glob('launch/*'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='augustmenard',
    maintainer_email='bmenard3@gatech.edu',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'moveRobot = team_rocket_final.moveRobot:main',
            'getObjectRange = team_rocket_final.getObjectRange:main'
        ],
    },
)
