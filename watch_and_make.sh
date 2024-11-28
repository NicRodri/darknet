#!/bin/bash

# Define paths relative to the script's location
BASE_DIR=$(dirname "$(readlink -f "$0")")  # Resolve the directory containing the script
SRC_DIR="$BASE_DIR/src"
WATCH_FILE="$SRC_DIR/detection_handler.cpp"

LOG_FILE="$BASE_DIR/watch_and_make.log"  # Define the log file path

# Clear the log file at the start of the script
> "$LOG_FILE"

echo "Watching $WATCH_FILE for changes..." | tee -a "$LOG_FILE"

LAST_MOD_TIME=""

# Define ANSI color codes
GREEN='\033[0;32m'
NC='\033[0m' # No color

while true; do
    # Get the last modified time of the file
    CURRENT_MOD_TIME=$(stat -c %Y "$WATCH_FILE")

    if [[ "$CURRENT_MOD_TIME" != "$LAST_MOD_TIME" ]]; then
        echo "Change detected in $WATCH_FILE. Running make clean and make..." | tee -a "$LOG_FILE"
        LAST_MOD_TIME="$CURRENT_MOD_TIME"
        make clean | tee -a "$LOG_FILE"
        make | tee -a "$LOG_FILE"
        echo -e "${GREEN}Build completed.${NC}" | tee -a "$LOG_FILE"
    fi

    sleep 1  # Check for changes every second
done
