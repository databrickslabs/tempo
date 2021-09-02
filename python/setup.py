import setuptools
from setuptools import find_packages

with open('./README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='dbl-tempo',
    version='0.1.1',
    author='Ricardo Portilla, Tristan Nixon, Max Thone, Sonali Guleria',
    author_email='labs@databricks.com',
    description='Spark Time Series Utility Package',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/databrickslabs/tempo',
    packages=find_packages(where=".", include=["tempo"]),
    extras_require=dict(tests=["pytest"]),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        ],
    )
