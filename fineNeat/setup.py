from setuptools import setup, find_packages

setup(
    name="fineNeat",
    version="0.2",
    packages=find_packages(),
    install_requires=[
        "matplotlib",
        "networkx",
        "numpy",
        "pillow",
        # Add other dependencies here
    ],
)