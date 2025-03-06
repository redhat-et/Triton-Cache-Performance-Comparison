#!/bin/bash

ARCH=""
TRITON_CACHE_DIR="$HOME/.triton/cache"
CUSTOM_SCRIPT="./scripts/flash_attention.py"
CUSTOM_TITLE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --arch)
            ARCH="$2"
            shift 2
            ;;
        --triton-cache-dir)
            TRITON_CACHE_DIR="$2"
            shift 2
            ;;
        --script)
            CUSTOM_SCRIPT="$2"
            shift 2
            ;;
        --title)
            CUSTOM_TITLE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "$ARCH" ]; then
    echo "Error: --arch must be specified (cuda or rocm)"
    exit 1
fi

if [ "$ARCH" != "cuda" ] && [ "$ARCH" != "rocm" ]; then
    echo "Error: --arch must be either 'cuda' or 'rocm'"
    exit 1
fi

LOG_FILE="gpu_usage_log.csv"
echo "session_id,timestamp,gpu_memory_used" > "$LOG_FILE"

log_usage() {
    local SESSION_ID=$1
    local START=$(date +%s)

    while true; do
        CURRENT_TIME=$(date +%s)
        ELAPSED_TIME=$((CURRENT_TIME - START))

        if [ "$ARCH" = "cuda" ]; then
            GPU_MEMORY_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
        else
            GPU_MEMORY_USED=$(rocm-smi | awk '$1 == "0" {print $(NF-1)}') # Change it to `n` if you want to use the n_th GPU
        fi

        echo "$SESSION_ID,$ELAPSED_TIME,$GPU_MEMORY_USED" >> "$LOG_FILE"
        sleep 1
    done
}

log_usage "no-cache" &
LOG_PID1=$!

echo "Removing Triton cache at $TRITON_CACHE_DIR..."
rm -rf "$TRITON_CACHE_DIR"

python "$CUSTOM_SCRIPT"

kill $LOG_PID1

log_usage "cache" &
LOG_PID2=$!

python "$CUSTOM_SCRIPT"

kill $LOG_PID2

echo "GPU memory usage data saved in $LOG_FILE"
echo "Saving the plot..."

if [ -z "$CUSTOM_TITLE" ]; then
    python ./scripts/plot_benchmark.py
else
    python ./scripts/plot_benchmark.py --title "$CUSTOM_TITLE"
fi
