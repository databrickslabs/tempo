import subprocess
import re
import semver


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
    semver.VersionInfo.parse(build_version)
    return build_version
