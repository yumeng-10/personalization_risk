#!/usr/bin/env bash
# Infinitely sends "hello" to the local llama3.1-70B sglang endpoint every second.
set -euo pipefail

URL="http://localhost:30005/v1/chat/completions"
PAYLOAD='{"model":"Llama-3.1-70B","messages":[{"role":"user","content":"hello"}],"max_tokens":16}'

i=0
while true; do
    i=$((i + 1))
    echo -n "[$(date '+%H:%M:%S')] #${i} "
    curl -s -o /dev/null -w "%{http_code} (%{time_total}s)\n" \
        -X POST "$URL" \
        -H "Content-Type: application/json" \
        -d "$PAYLOAD"
    sleep 1
done
