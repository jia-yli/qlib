#!/bin/bash
PGID=$$
trap "echo 'Ctrl-C caught, killing all workers'; kill -- -$PGID" INT

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

NUM_WORKERS=4

UTC_TIMESTAMP="$(date --utc +'%Y%m%d_%H%M%S')"
LOG_DIR="$SCRIPT_DIR/logs_${UTC_TIMESTAMP}_utc"
mkdir -p "$LOG_DIR"

echo "[${UTC_TIMESTAMP} UTC] Launching $NUM_WORKERS workers. Logs in: $LOG_DIR"

for i in $(seq 1 "$NUM_WORKERS"); do
  LOG_FILE="$LOG_DIR/worker_${i}_log.txt"
  python "$SCRIPT_DIR/tune.py" >>"$LOG_FILE" 2>&1 &
done

wait

UTC_TIMESTAMP="$(date --utc +'%Y%m%d_%H%M%S')"
echo "[${UTC_TIMESTAMP} UTC] All workers finished."

