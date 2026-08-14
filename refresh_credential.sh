#!/bin/bash
#
# Refresh ada credentials every 15 minutes, automatically stopping after 8 hours.
#
# Usage:
#   ./refresh_credentials.sh              # run in foreground
#   nohup ./refresh_credentials.sh > refresh_credentials.log 2>&1 &   # run in background
#

INTERVAL=$((15 * 60))          # 15 minutes, in seconds
DURATION=$((8 * 60 * 60))      # 8 hours, in seconds

CMD="ada credentials update --account=745184793497 --provider=conduit --role=IibsAdminAccess-DO-NOT-DELETE --once"

start_time=$(date +%s)
end_time=$((start_time + DURATION))

echo "Starting credential refresh loop at $(date)"
echo "Will stop at $(date -r "$end_time")"

while [ "$(date +%s)" -lt "$end_time" ]; do
    echo "[$(date)] Running: $CMD"
    $CMD

    # Stop if the next run would fall after the end time.
    if [ "$(( $(date +%s) + INTERVAL ))" -ge "$end_time" ]; then
        break
    fi

    sleep "$INTERVAL"
done

echo "Reached 8-hour limit. Stopping at $(date)"
