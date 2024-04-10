from setuptools import setup, find_packages
import io
import os
from pathlib import Path



setup(
    name='deeplearning_package',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        # List dependencies here
        'numpy',
        'tensorflow',  # or 'torch' for PyTorch
        # Add other dependencies as needed
    ],
)

# Metadata of package
NAME = 'deeplearning_package'
DESCRIPTION = 'Rain Prediction Model using artificial neural network'
EMAIL = 'deepchandoa@gmail.com'
AUTHOR = 'Deepchand O A'
REQUIRES_PYTHON = '>=3.8.0'

pwd = os.path.abspath(os.path.dirname(__file__))

# Get the list of packages to be installed
def list_reqs(fname='requirements.txt'):
    with io.open(os.path.join(pwd, fname), encoding='utf-8') as f:
        return f.read().splitlines()

try:
    with io.open(os.path.join(pwd, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

ROOT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = ROOT_DIR 
about = {}
with open(PACKAGE_DIR / 'VERSION') as f:
    _version = f.read().strip()
    about['__version__'] = _version


setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    packages=find_packages(exclude=('tests',)),
    package_data={'deeplearning_package': ['VERSION']},
    install_requires=list_reqs(),
    extras_require={},
    include_package_data=True,
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
)