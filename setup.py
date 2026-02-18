import os
from setuptools import find_packages, setup

package_name = 'laser_uav_missions'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'models'), ['models/gesture_recognizer.task']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='erick',
    maintainer_email='hiagoe121@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'gesture_control = laser_uav_missions.gesture_control:main'
        ],
    },
)
