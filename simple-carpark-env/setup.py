from setuptools import setup, find_packages

setup(
    name="simple_carpark",
    version='0.0.1',
    install_requires=[
        'gym',
        'pybullet',
        'numpy',
        'matplotlib',
        'torch'
    ],
    packages=find_packages(include=["simple_carpark", "simple_carpark.*"]),
    package_data={'simple_carpark': ['resources/*.urdf']}
)