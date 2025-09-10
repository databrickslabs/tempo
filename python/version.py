import subprocess

CURRENT_VERSION = "0.1.30"
# run a shell command and return stdout
def run_cmd(cmd):
    cmd_proc = subprocess.run(cmd, shell=True, capture_output=True)
    if cmd_proc.returncode != 0:
        raise OSError(
            f"Shell command '{cmd}' failed with return code {cmd_proc.returncode}\n"
            f"STDERR: {cmd_proc.stderr.decode('utf-8')}"
        )
    return cmd_proc.stdout.decode("utf-8").strip()


# fetch the most recent build version for hatch environment creation
def get_version():
    """Return the package version based on latest git tag."""
    import os

    # Optionally override with environment variable
    if "PACKAGE_VERSION" in os.environ:
        return os.environ.get("PACKAGE_VERSION")

    # Fall back to git tag
    try:
        return CURRENT_VERSION
    except (OSError, ValueError) as E:
        # Return a fallback version if git operations fail
        # TODO - Raise with error message
        raise E


__version__ = get_version()
