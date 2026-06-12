#!/usr/bin/env bash
# Pull experiment results from the training box into local results/.
#
# Usage:
#   ./scripts/pull_results.sh          one-shot pull
#   ./scripts/pull_results.sh --loop   pull every 30 min; after the program
#                                      completes and a final pull succeeds,
#                                      touch ~/results_pulled on the box so
#                                      the watchdog shuts it down; exit when
#                                      the box disappears (8 failed attempts).
set -uo pipefail
BOX=ubuntu@44.200.193.172
KEY="$HOME/.ssh/devnet-train-key.pem"
SSH_CMD="ssh -i $KEY -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

pull_once() {
  rsync -az -e "$SSH_CMD" --exclude mallanet_baseline \
    "$BOX:mallanet/results/" "$ROOT/results/" || return 1
  $SSH_CMD "$BOX" 'cat ~/program.log' > "$ROOT/results/program.log" || return 1
}

if [ "${1:-}" != "--loop" ]; then
  pull_once && echo "$(date -u) pull ok"
  exit $?
fi

fails=0
while true; do
  if pull_once; then
    fails=0
    echo "$(date -u) pull ok"
    if grep -q PROGRAM_COMPLETE "$ROOT/results/program.log" 2>/dev/null; then
      $SSH_CMD "$BOX" 'touch ~/results_pulled'
      echo "$(date -u) PROGRAM_COMPLETE: final pull done, shutdown marker set"
      exit 0
    fi
  else
    fails=$((fails + 1))
    echo "$(date -u) pull failed ($fails/8)"
    [ "$fails" -ge 8 ] && { echo "$(date -u) box unreachable, exiting"; exit 1; }
  fi
  sleep 1800
done
