#!/bin/bash
#
# Run absolutely everything: all 15 PAT distillation jobs followed by all 5
# standalone baselines. Sequential. No arguments. Just go.
#
# Usage:
#   bash run_everything.sh
#
# Each leg's failures are reported in its own summary. Set STOP_ON_ERROR=1
# (forwarded to both run_all.sh scripts) to abort on first failure.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

START="$(date)"

echo "════════════════════════════════════════════════════════════"
echo " RUN EVERYTHING"
echo " 15 PAT distillation jobs + 5 standalone baselines"
echo " Started: ${START}"
echo "════════════════════════════════════════════════════════════"
echo ""

PAT_RC=0
STANDALONE_RC=0

echo "▶ Phase 1/2: PAT distillation"
echo ""
bash "${SCRIPT_DIR}/run_all.sh" || PAT_RC=$?

echo ""
echo "▶ Phase 2/2: Standalone baselines"
echo ""
( cd "${SCRIPT_DIR}/standalone_training" && bash run_all.sh ) || STANDALONE_RC=$?

echo ""
echo "════════════════════════════════════════════════════════════"
echo " Started:  ${START}"
echo " Finished: $(date)"
echo " PAT phase exit:        ${PAT_RC}"
echo " Standalone phase exit: ${STANDALONE_RC}"
echo "════════════════════════════════════════════════════════════"

if [ "${PAT_RC}" -ne 0 ] || [ "${STANDALONE_RC}" -ne 0 ]; then
    exit 1
fi
