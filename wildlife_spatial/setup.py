from setuptools import find_packages, setup

package_name = 'wildlife_spatial'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', [
            'config/static_large_demo_ground_truth.json',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='tony',
    maintainer_email='HongLinh.Nguyen@student.uts.edu.au',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'candidate_localizer = wildlife_spatial.candidate_localizer:main',
            'candidate_registry = wildlife_spatial.candidate_registry:main',
            'spatial_evaluator = wildlife_spatial.spatial_evaluator:main',
        ],
    },
)
