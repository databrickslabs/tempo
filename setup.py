import setuptools
from setuptools import find_packages

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='tca',
    version='0.1',
    author='Ricardo Portilla',
    author_email='ricardo.portilla@databricks.com',
    description='scalable python time series utility package',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/rportilla-databricks/tsfinancial',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        ],
    )
