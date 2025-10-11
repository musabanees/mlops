from setuptools import find_packages, setup

setup(
    name='mlops_tutorial',
    version='0.0.0',
    author="Masab",
    author_email="masaba019@gmail.com",
    packages=find_packages(where='src'),
    package_dir={"": "src"},
    install_requires=[]
)