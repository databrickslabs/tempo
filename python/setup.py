import setuptools
from setuptools import find_packages

from libversioner import get_version

with open('./README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='dbl-tempo',
    version=get_version('dbl-tempo', major=False, minor=True, micro=False, env='PROD'),
    author='Ricardo Portilla, Tristan Nixon, Max Thone, Sonali Guleria',
    author_email='labs@databricks.com',
    description='Spark Time Series Utility Package',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/databrickslabs/tempo',
    packages=find_packages(where=".", include=["tempo"]),
    install_requires=[
     'ipython',
     'pandas',
     'scipy'
    ],
    extras_require=dict(tests=["pytest"]),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: Other/Proprietary License',
        'Operating System :: OS Independent',
        ],
    )
