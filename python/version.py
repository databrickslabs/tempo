CURRENT_VERSION = "0.1.30"


def get_version():
    """Return the package version based on CURRENT_VERSION or env override."""
    import os

    # Optionally override with environment variable
    if "PACKAGE_VERSION" in os.environ:
        return os.environ.get("PACKAGE_VERSION")

    return CURRENT_VERSION


__version__ = get_version()
