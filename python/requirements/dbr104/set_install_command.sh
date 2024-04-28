#!/bin/bash
# Based on architecture, set an appropriate install command
SYS_ARCH=$(uname -m)
export SYS_ARCH
if [ "$SYS_ARCH" = "arm64" ]; then
    NO_BINARY_PACKAGES="pyarrow,pandas,scipy"
    echo "pip install --no-binary $NO_BINARY_PACKAGES {opts} {packages}" > install_cmd.txt
else
    echo "pip install {opts} {packages}" > install_cmd.txt
fi
