from setuptools import find_packages, setup
from version import get_latest_git_tag

# fetch the most recent version tag to use as build version
build_version = get_latest_git_tag()

# use the contents of the README file as the 'long description' for the package
with open("./README.md", "r") as fh:
    long_description = fh.read()

#
# build the package
#
setup(
    name="dbl-tempo",
    version=build_version,
    author="Ricardo Portilla, Tristan Nixon, Max Thone, Sonali Guleria",
    author_email="labs@databricks.com",
    description="Spark Time Series Utility Package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://databrickslabs.github.io/tempo/",
    packages=find_packages(where=".", include=["tempo"]),
    extras_require=dict(tests=["pytest"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
    ],
)
