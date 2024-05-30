#!/usr/bin/env bash

# Check if both source and destination directory arguments are provided
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 /path/to/source_directory /path/to/destination_directory"
    exit 1
fi

# Define source and destination directories from arguments
SOURCE_DIR="$1"
DEST_DIR="$2"

# Check if the source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Source directory does not exist: $SOURCE_DIR"
    exit 1
fi

# Check if the destination directory exists, if not, create it
if [ ! -d "$DEST_DIR" ]; then
    mkdir -p "$DEST_DIR"
fi

# Move all files from source to destination
mv "$SOURCE_DIR"/* "$DEST_DIR"
