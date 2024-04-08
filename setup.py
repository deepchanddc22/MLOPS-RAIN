from setuptools import setup, find_packages

setup(
    name='deeplearning_package',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        # List dependencies here
        'numpy',
        'tensorflow',  # or 'torch' for PyTorch
        # Add other dependencies as needed
    ],
)
