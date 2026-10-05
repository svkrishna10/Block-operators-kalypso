#!/usr/bin/env bash

set -o pipefail

NAME="$1"
shift

LOG_DIR="experiments/logs"

mkdir -p "$LOG_DIR"

LOG="$LOG_DIR/${NAME}.log"
BEFORE="$LOG_DIR/${NAME}-metrics-before.prom"
AFTER="$LOG_DIR/${NAME}-metrics-after.prom"
TIMESERIES="$LOG_DIR/${NAME}-metrics-timeseries.prom"
MARKER="$LOG_DIR/${NAME}.sampling"

echo "=================================================="
echo "Running: $NAME"
echo "=================================================="

echo "Saving metrics before run..."
curl -s http://localhost:8003/metrics > "$BEFORE"

touch "$MARKER"

(
    while [ -f "$MARKER" ]; do
        echo "# scrape_time $(date +%s.%N)"
        curl -s http://localhost:8003/metrics
        echo
        sleep 1
    done
) > "$TIMESERIES" 2>/dev/null &

SAMPLER_PID=$!

echo "Running workload..."

/usr/bin/time -v \
    "$@" \
    2>&1 | tee "$LOG"

RC=${PIPESTATUS[0]}

rm -f "$MARKER"
wait "$SAMPLER_PID" 2>/dev/null || true

echo "Saving metrics after run..."
curl -s http://localhost:8003/metrics > "$AFTER"

echo
echo "Run: $NAME"
echo "Exit code: $RC"
echo "Log: $LOG"
echo "Metrics before: $BEFORE"
echo "Metrics after:  $AFTER"
echo "Metrics timeseries: $TIMESERIES"

exit "$RC"
