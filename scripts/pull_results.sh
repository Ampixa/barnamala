#!/usr/bin/env bash
# Pull experiment results from the training box into local results/.
#
# Usage:
#   ./scripts/pull_results.sh          one-shot pull
#   ./scripts/pull_results.sh --loop   pull every 30 min; after the run's
#                                      completion marker appears and a final
#                                      pull succeeds, touch ~/results_pulled on
#                                      the box so the watchdog shuts it down;
#                                      exit when the box disappears (8 fails).
#
# Completion marker is parameterized for chained runs:
#   MARKER       completion string to wait for  (default PROGRAM_COMPLETE)
#   COMPLETE_LOG log file holding it            (default program.log)
# e.g. MARKER=PROGRAM2_COMPLETE COMPLETE_LOG=program2.log ./pull_results.sh --loop
set -uo pipefail
# Set BOX and KEY for your training host, e.g.:
#   BOX=ubuntu@<your-ec2-public-ip> KEY=~/.ssh/<your-key>.pem ./scripts/pull_results.sh
BOX="${BOX:?set BOX=user@host for your training box}"
KEY="${KEY:-$HOME/.ssh/id_rsa}"
SSH_CMD="ssh -i $KEY -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MARKER="${MARKER:-PROGRAM_COMPLETE}"
COMPLETE_LOG="${COMPLETE_LOG:-program.log}"

pull_once() {
  rsync -az -e "$SSH_CMD" --exclude mallanet_baseline \
    "$BOX:mallanet/results/" "$ROOT/results/" || return 1
  # pull both logs; program2.log may not exist yet (phase 2 not started)
  $SSH_CMD "$BOX" 'cat ~/program.log 2>/dev/null; \
    [ -f ~/program2.log ] && { echo; cat ~/program2.log; } || true' \
    > "$ROOT/results/program.log" || return 1
  $SSH_CMD "$BOX" 'cat ~/program2.log 2>/dev/null || true' \
    > "$ROOT/results/program2.log" || true
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
    if grep -q "$MARKER" "$ROOT/results/$COMPLETE_LOG" 2>/dev/null; then
      $SSH_CMD "$BOX" 'touch ~/results_pulled'
      echo "$(date -u) $MARKER: final pull done, shutdown marker set"
      exit 0
    fi
  else
    fails=$((fails + 1))
    echo "$(date -u) pull failed ($fails/8)"
    [ "$fails" -ge 8 ] && { echo "$(date -u) box unreachable, exiting"; exit 1; }
  fi
  sleep 1800
done
