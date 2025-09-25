from setuptools import find_packages, setup

package_name = 'team_rocket_nodes'

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
    maintainer='michaelangelo',
    maintainer_email='marilynbraojos@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'chase_object = team_rocket_nodes.chase_object_node:main',
            'detect_object = team_rocket_nodes.detect_object_node:main',
            'get_object_range = team_rocket_nodes.get_object_range_node:main',
        ],
    },

)
