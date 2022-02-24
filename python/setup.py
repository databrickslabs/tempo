import setuptools
from setuptools import find_packages

with open('./README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='dbl-tempo',
    version='0.1.8',
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
