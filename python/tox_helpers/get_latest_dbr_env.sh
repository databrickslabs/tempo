#!/usr/bin/env bash

# Exit script if any command fails
set -e

# Check if tox is installed
if ! command -v tox &> /dev/null; then
    echo "tox could not be found. Please install it and try again."
    exit 1
fi

# Get the most recent dbr environment from tox (lexicographically sorted)
LATEST_DBR_TOX_ENV=$(tox -l | grep dbr | tail -n 1)

# Check if we got a valid environment
if [ -z "$LATEST_DBR_TOX_ENV" ]; then
    echo "No dbr environments found in tox."
    exit 1
fi

# Define the directory and file path
DIR="temp"
FILE="$DIR/.env"

# Create the directory if it doesn't exist
# Check if the directory exists, and create it if it doesn't
if [ ! -d "$DIR" ]; then
    mkdir -p "$DIR"
    echo "Directory $DIR created."
else
    echo "Directory $DIR already exists."
fi

# Ensure the directory exists
if [ ! -d "$DIR" ]; then
    echo "Directory $DIR could not be created. Exiting."
    exit 1
fi

# Write the value to the file (this will also create the file if it doesn't exist)
echo "LATEST_DBR_TOX_ENV=$LATEST_DBR_TOX_ENV" > "$FILE"

# Check if the file was created and written successfully
if [ ! -f "$FILE" ]; then
    echo "File $FILE could not be created or written to. Exiting."
    exit 1
fi

# Optionally, print the value if the -v flag is provided
echo "LATEST_DBR_TOX_ENV has been written to $FILE with value: $LATEST_DBR_TOX_ENV"
