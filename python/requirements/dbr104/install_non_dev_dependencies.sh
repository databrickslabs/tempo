#!/usr/bin/env bash

set -e  # Exit on error
[ -n "$DEBUG" ] && set -x  # Enable debugging if DEBUG environment variable is set

# This runs from the root of the repository
ARM_REQ_FILE="$(pwd)/requirements/dbr104/dbr104_arm.txt"
GENERIC_REQ_FILE="$(pwd)/requirements/dbr104/dbr104.txt"

# Check necessary commands and files
command -v pip >/dev/null 2>&1 || { echo >&2 "pip is required but it's not installed. Aborting."; exit 1; }
[ -f "$ARM_REQ_FILE" ] || { echo >&2 "Required file $ARM_REQ_FILE not found. Aborting."; exit 1; }
[ -f "$GENERIC_REQ_FILE" ] || { echo >&2 "Required file $GENERIC_REQ_FILE not found. Aborting."; exit 1; }

# Get the architecture of the system
sys_arch=$(uname -m)
echo "System Architecture: $sys_arch"

echo "Upgrading pip..."
pip install --upgrade pip

case "$sys_arch" in
    arm*)
        echo "ARM Architecture detected. Specific model: $sys_arch"
        echo "Installing ARM-specific dependencies..."
        pip install -r "$ARM_REQ_FILE"
        pip install --no-deps pandas~=1.2.4
        pip install --no-deps pyarrow~=4.0.0
        pip install --no-deps scipy~=1.6.2
        ;;
    *)
        echo "Non-ARM Architecture: $sys_arch"
        echo "Installing generic dependencies..."
        pip install -r "$GENERIC_REQ_FILE"
        ;;
esac
