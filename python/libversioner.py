import json

import requests

try:
    from packaging.version import parse
except ImportError:
    from pip._vendor.packaging.version import parse


def get_version(package: str, major: bool = False, minor: bool = False, micro: bool = False) -> str:
    """
    This function is for getting the version of package from PyPI and generate a updated version number
    for the new build. Or an initial version number if the package is being released for the first time
    to PyPI (version in this cases will be 0.0.0).

    This method expects versions to follow standard convention:

    `sample-package version major.minor.micro`

    For example:

    `dbl-tempo version 0.1.7`

    * 0: MAJOR_VERSION_NUMBER. This is updated when the package receives the big update that likely distinguishes it from the previous releases.
    * 1: MINOR_VERSION_NUMBER. This is updated when new features are launched in the existing release of the package.
    * 7: MICRO_VERSION_NUMBER. This is updated when bug fixes or optimizations are released for the package.

    Returns updated version of package for pypi.python.org i.e in this case the function returns 0.2.7 if minor flag is true.

    :param package: Name of the package
    :type package: string
    :param major: MAJOR VERSION NUMBER Flag
    :type major: boolean
    :param minor: MINOR VERSION NUMBER Flag
    :type minor: boolean
    :param micro: MICRO VERSION NUMBER Flag
    :type micro: boolean
    :return: The updated version number
    :rtype: string
    """
    # API endpoint of PyPI
    url_pattern = 'https://pypi.python.org/pypi/{package}/json'
    # Getting our package specific details
    req = requests.get(url_pattern.format(package=package))
    # Initializing a zero version
    version = parse('0')
    # Initializing new major, minor & micro
    new_major, new_minor, new_micro = version.major, version.minor, version.micro
    # Condition to check if this is a existing package or not
    if req.status_code == requests.codes.ok:
        # Converting the response to dictionary using the correct encoding
        j = json.loads(req.text.encode(req.encoding or "utf-8"))
        # Getting the release information of the package from the dictionary
        releases = j.get('releases', [])
        # Iterating through the release information to get the latest version
        for release in releases:
            ver = parse(release)
            if not ver.is_prerelease:
                version = max(version, ver)
        # Now updating the latest version based on the type of updates to be performed
        if major:
            new_major = version.major + 1
        else:
            new_major = version.major

        if minor:
            new_minor = version.minor + 1
        else:
            new_minor = version.minor

        if micro:
            new_micro = version.micro + 1
        else:
            new_micro = version.micro
        # Constructing the new updated version number
        new_version = f'{new_major}.{new_minor}.{new_micro}'
        # Returning the new version number
        return new_version
    else:
        # Returning initial version number for new package
        return f'{new_major}.{new_minor}.{new_micro}'
