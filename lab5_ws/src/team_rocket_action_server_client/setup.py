from setuptools import find_packages, setup

package_name = 'team_rocket_action_server_client'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
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
            'action_client = team_rocket_action_server_client.action_client:main',
            'waypoint_navigator = team_rocket_action_server_client.waypoint_navigator:main'
        ],
    },
)
