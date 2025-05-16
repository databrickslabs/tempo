import subprocess
import re


# run a shell command and return stdout
def run_cmd(cmd):
    cmd_proc = subprocess.run(cmd, shell=True, capture_output=True)
    if cmd_proc.returncode != 0:
        raise OSError(
            f"Shell command '{cmd}' failed with return code {cmd_proc.returncode}\n"
            f"STDERR: {cmd_proc.stderr.decode('utf-8')}"
        )
    return cmd_proc.stdout.decode("utf-8").strip()


# fetch the most recent version tag to use as build version
def get_latest_git_tag():
    latest_tag = run_cmd("git describe --abbrev=0 --tags")
    build_version = re.sub("v\.?\s*", "", latest_tag)
    # validate that this is a valid semantic version - will throw exception if not
    try:
        import semver
        semver.VersionInfo.parse(build_version)
    except (ModuleNotFoundError) as module_not_found_error:
        # unable to validate because semver is not installed in barebones env for hatch
        pass
    return build_version


# fetch the most recent build version for hatch environment creation
def get_version():
    """Return the package version based on latest git tag."""
    import os
    
    # Optionally override with environment variable
    if "PACKAGE_VERSION" in os.environ:
        return os.environ.get("PACKAGE_VERSION")
    
    # Fall back to git tag
    try:
        return get_latest_git_tag()
    except (OSError, ValueError) as E:
        # Return a fallback version if git operations fail
        # TODO - Raise with error message
        raise E
    

__version__ = get_version()