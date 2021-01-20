from setuptools import setup
from pathlib import Path

THISDIR = Path(__file__).parent

with open(THISDIR/'requirements.txt') as f:
    required = f.read().splitlines()

with open(THISDIR/'README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="sympy-nondim",
    author='Christoph Heindl',
    description='Non-dimensionalization of physical variables in sympy', 
    url='https://github.com/cheind/sympy-nondim/',
    license='MIT',
    long_description=long_description,
    long_description_content_type='text/markdown',
    version=open(THISDIR/'nondim'/'__init__.py').readlines()[-1].split()[-1].strip('\''),
    packages=['nondim', 'nondim.tests'],    
    install_requires=required,
    zip_safe=False,
)