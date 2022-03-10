import subprocess
import re
from setuptools import find_packages, setup
import semver


# run a shell command and return stdout
def run_cmd(cmd):
    cmd_proc = subprocess.run(cmd, shell=True, capture_output=True)
    if cmd_proc.returncode != 0:
        raise OSError(f"Shell command '{cmd}' failed with return code {cmd_proc.returncode}\n"
                      f"STDERR: {cmd_proc.stderr.decode('utf-8')}")
    return cmd_proc.stdout.decode('utf-8').strip()


#
# fetch the most recent version tag to use as build version
#
latest_tag = run_cmd('git describe --abbrev=0 --tags')
build_version = re.sub('v\.?\s*', '', latest_tag)
# validate that this is a valid semantic version - will throw exception if not
semver.VersionInfo.parse(build_version)

# use the contents of the README file as the 'long description' for the package
with open('./README.md', 'r') as fh:
    long_description = fh.read()

#
# build the package
#
setup(
    name='dbl-tempo',
    version=build_version,
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
