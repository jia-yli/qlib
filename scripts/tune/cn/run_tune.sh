#!/bin/bash
PGID=$$
trap "echo 'Ctrl-C caught, killing all workers'; kill -- -$PGID" INT

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

SUFFIX="${1:-single}"

NUM_WORKERS=4
NUM_GPUS=4

UTC_TIMESTAMP="$(date --utc +'%Y%m%d_%H%M%S')"
HOSTNAME="$(hostname -s)"
LOG_DIR="$SCRIPT_DIR/logs/logs_${SUFFIX}_${UTC_TIMESTAMP}_utc_${HOSTNAME}"
mkdir -p "$LOG_DIR"

echo "[${UTC_TIMESTAMP} UTC] Launching $NUM_WORKERS workers. Logs in: $LOG_DIR"

for i in $(seq 0 $((NUM_WORKERS - 1))); do
  LOG_FILE="$LOG_DIR/worker_${i}_log.txt"
  CUDA_VISIBLE_DEVICES="$(( i % NUM_GPUS ))" \
  python "${SCRIPT_DIR}/tune_${SUFFIX}.py" >>"$LOG_FILE" 2>&1 &
done

wait

UTC_TIMESTAMP="$(date --utc +'%Y%m%d_%H%M%S')"
echo "[${UTC_TIMESTAMP} UTC] All workers finished."

